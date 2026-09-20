"""Spatiotemporal panels and realistic masks for ImputeFormer training.

The original dataset exposes one NetCDF file per station and variable.  This
module aligns those files on a common hourly axis so a model can learn from
both temporal and cross-station structure.  Missing auxiliary products are
represented by an explicit availability flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from giano.evaluation.gapfill.masks import block_mask, point_mask
from giano.meteorology import (
    derive_wind_direction_from_uv,
    derive_wind_speed_from_uv,
)
from giano.netcdf import open_dataset_robust

MaskMode = Literal[
    "mixed",
    "point",
    "block",
    "spatial_block",
    "terminal",
    "empirical",
    "natural",
]
TemporalSplit = Literal["train", "val", "test", "all"]

INPUT_FEATURES = (
    "station_value",
    "temporal_baseline",
    "auxiliary_value",
    "station_observed",
    "auxiliary_observed",
    "baseline_available",
    "hour_sin",
    "hour_cos",
    "year_sin",
    "year_cos",
    "direction_baseline_sin",
    "direction_baseline_cos",
)

_NS_PER_HOUR = 3_600_000_000_000
_SCALE_FLOORS = {
    "temperature": 1.0,
    "precipitation": 0.25,
    "humidity": 5.0,
    "pressure": 1.0,
    "wind_speed": 0.5,
    "wind_direction": 180.0,
}


@dataclass(frozen=True)
class SpatiotemporalPanel:
    """Aligned values for one meteorological variable."""

    variable: str
    times: np.ndarray
    station_ids: tuple[str, ...]
    coordinates: np.ndarray
    values: np.ndarray
    auxiliary: np.ndarray | None
    coverage_bounds: np.ndarray
    source_paths: tuple[Path, ...]

    @property
    def num_nodes(self) -> int:
        """Return the number of stations."""
        return len(self.station_ids)


def _station_id(path: Path) -> str:
    return path.stem.split("_", maxsplit=1)[0]


def _coordinates(ds: xr.Dataset) -> tuple[float, float]:
    def first(names: tuple[str, ...]) -> float | None:
        for name in names:
            if name in ds.coords:
                value = np.asarray(ds.coords[name].values, dtype=float).reshape(-1)
                if value.size and np.isfinite(value[0]):
                    return float(value[0])
            if name in ds.attrs:
                try:
                    attr_value = float(ds.attrs[name])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(attr_value):
                    return attr_value
        return None

    latitude = first(("latitude", "lat", "station_latitude"))
    longitude = first(("longitude", "lon", "station_longitude"))
    if latitude is None or longitude is None:
        raise ValueError("Processed station file lacks latitude/longitude")
    return float(np.clip(latitude, -90, 90)), float(np.clip(longitude, -180, 180))


def _auxiliary_values(ds: xr.Dataset, variable: str) -> np.ndarray | None:
    if variable in ds.data_vars and variable != "value":
        if ds[variable].dims != ds["value"].dims:
            raise ValueError(f"Auxiliary {variable} is not aligned with value")
        return np.asarray(ds[variable].values, dtype=np.float32)
    if {"wind_u", "wind_v"}.issubset(ds.data_vars):
        if (
            ds["wind_u"].dims != ds["value"].dims
            or ds["wind_v"].dims != ds["value"].dims
        ):
            raise ValueError("Auxiliary wind components are not aligned with value")
        wind_u = np.asarray(ds["wind_u"].values, dtype=np.float32)
        wind_v = np.asarray(ds["wind_v"].values, dtype=np.float32)
        if variable == "wind_speed":
            return derive_wind_speed_from_uv(wind_u, wind_v).astype(np.float32)
        if variable == "wind_direction":
            return derive_wind_direction_from_uv(wind_u, wind_v).astype(np.float32)
    return None


def build_spatiotemporal_panel(
    data_root: str | Path,
    variable: str,
) -> SpatiotemporalPanel:
    """Load and align every station available for ``variable``.

    Existing station-disjoint folders are pooled deliberately.  Train,
    validation, and test separation is performed later along the time axis,
    which matches the operational task of repairing gaps in known stations.
    """
    root = Path(data_root)
    paths = sorted(root.glob(f"*/*_{variable}_merged.nc"))
    if not paths:
        raise FileNotFoundError(f"No processed files found for {variable} in {root}")

    records: list[tuple[Path, np.ndarray, tuple[float, float], bool]] = []
    seen_stations: set[str] = set()
    first_time: np.datetime64 | None = None
    last_time: np.datetime64 | None = None
    any_auxiliary = False
    for path in paths:
        station = _station_id(path)
        if station in seen_stations:
            raise ValueError(f"Duplicate processed station for {variable}: {station}")
        seen_stations.add(station)
        with open_dataset_robust(path) as ds:
            if "time" not in ds.coords or "value" not in ds.data_vars:
                raise ValueError(f"Invalid processed contract: {path}")
            times = np.asarray(ds["time"].values).astype("datetime64[ns]")
            if times.ndim != 1 or not times.size:
                raise ValueError(f"Empty or invalid time axis: {path}")
            if np.any(np.diff(times.astype(np.int64)) <= 0):
                raise ValueError(f"Time axis is not strictly increasing: {path}")
            has_auxiliary = _auxiliary_values(ds, variable) is not None
            records.append((path, times, _coordinates(ds), has_auxiliary))
            any_auxiliary = any_auxiliary or has_auxiliary
            start = times[0]
            end = times[-1]
            first_time = start if first_time is None else min(first_time, start)
            last_time = end if last_time is None else max(last_time, end)

    if first_time is None or last_time is None:
        raise RuntimeError(f"Could not determine time range for {variable}")
    start_ns = int(first_time.astype(np.int64))
    end_ns = int(last_time.astype(np.int64))
    length = (end_ns - start_ns) // _NS_PER_HOUR + 1
    if length <= 0:
        raise ValueError(f"Invalid panel duration for {variable}")

    values = np.full((length, len(records)), np.nan, dtype=np.float32)
    auxiliary = (
        np.full_like(values, np.nan, dtype=np.float32) if any_auxiliary else None
    )
    coordinates = np.zeros((len(records), 2), dtype=np.float32)
    coverage_bounds = np.zeros((len(records), 2), dtype=np.int64)
    station_ids: list[str] = []
    source_paths: list[Path] = []

    for node, (path, expected_times, coords, _has_auxiliary) in enumerate(records):
        with open_dataset_robust(path) as ds:
            times_ns = expected_times.astype(np.int64)
            offsets, remainders = np.divmod(times_ns - start_ns, _NS_PER_HOUR)
            if np.any(remainders != 0):
                raise ValueError(f"Non-hourly timestamps in {path}")
            station_values = np.asarray(ds["value"].values, dtype=np.float32)
            if station_values.shape != offsets.shape:
                raise ValueError(f"Time/value length mismatch in {path}")
            values[offsets, node] = station_values
            coverage_bounds[node] = (int(offsets[0]), int(offsets[-1]))
            if auxiliary is not None:
                aux_values = _auxiliary_values(ds, variable)
                if aux_values is not None:
                    if aux_values.shape != offsets.shape:
                        raise ValueError(f"Time/auxiliary length mismatch in {path}")
                    auxiliary[offsets, node] = aux_values
            coordinates[node] = coords
            station_ids.append(_station_id(path))
            source_paths.append(path)

    times = np.arange(length, dtype=np.int64) * _NS_PER_HOUR + start_ns
    return SpatiotemporalPanel(
        variable=variable,
        times=times.astype("datetime64[ns]"),
        station_ids=tuple(station_ids),
        coordinates=coordinates,
        values=values,
        auxiliary=auxiliary,
        coverage_bounds=coverage_bounds,
        source_paths=tuple(source_paths),
    )


def _circular_difference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return ((values - reference + 180.0) % 360.0) - 180.0


def _temporal_baseline(values: np.ndarray, variable: str) -> np.ndarray:
    positions = np.arange(values.shape[0])
    baseline = np.full_like(values, np.nan, dtype=np.float32)
    for node in range(values.shape[1]):
        observed = np.isfinite(values[:, node])
        observed_positions = np.flatnonzero(observed)
        if not observed_positions.size:
            continue
        observed_values = values[observed, node]
        if variable == "wind_direction":
            unwrapped = np.rad2deg(np.unwrap(np.deg2rad(observed_values)))
            wrapped = np.mod(
                np.interp(positions, observed_positions, unwrapped),
                360.0,
            )
            baseline[:, node] = np.where(
                np.isclose(wrapped, 360.0, atol=1e-5),
                0.0,
                wrapped,
            )
        else:
            baseline[:, node] = np.interp(
                positions,
                observed_positions,
                observed_values,
            )
    return baseline


class SpatiotemporalWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Window a panel and create point, temporal-block, or spatial-block masks."""

    def __init__(  # noqa: PLR0913
        self,
        panel: SpatiotemporalPanel,
        split: TemporalSplit,
        *,
        seq_len: int = 72,
        stride: int = 12,
        max_nodes: int = 24,
        min_context_points: int = 4,
        seed: int = 42,
        mask_mode: MaskMode = "mixed",
        point_rate: tuple[float, float] = (0.15, 0.45),
        block_lengths: tuple[int, ...] = (3, 6, 12, 24),
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        required_node_indices: tuple[int, ...] = (),
    ) -> None:
        if seq_len < 4:
            raise ValueError("seq_len must be at least 4")
        if stride <= 0 or max_nodes <= 0 or min_context_points < 2:
            raise ValueError("stride/max_nodes/min_context_points are invalid")
        if not np.isclose(sum(split_ratios), 1.0):
            raise ValueError("split_ratios must sum to one")
        if mask_mode not in {
            "mixed",
            "point",
            "block",
            "spatial_block",
            "terminal",
            "empirical",
            "natural",
        }:
            raise ValueError(f"Unsupported mask mode: {mask_mode}")
        if not 0 < point_rate[0] <= point_rate[1] < 1:
            raise ValueError("point_rate must be inside (0, 1)")
        valid_blocks = tuple(int(v) for v in block_lengths if 0 < v < seq_len)
        if mask_mode != "empirical":
            valid_blocks = tuple(sorted(set(valid_blocks)))
        if not valid_blocks:
            raise ValueError("block_lengths must contain a value smaller than seq_len")

        self.panel = panel
        self.split = split
        self.seq_len = seq_len
        self.max_nodes = min(max_nodes, max(1, panel.num_nodes))
        self.min_context_points = min_context_points
        self.seed = seed
        self.epoch = 0
        self.mask_mode = mask_mode
        self.point_rate = point_rate
        self.block_lengths = valid_blocks
        if any(not 0 <= node < panel.num_nodes for node in required_node_indices):
            raise ValueError("required node index lies outside the panel")
        if len(set(required_node_indices)) != len(required_node_indices):
            raise ValueError("required node indices must be unique")
        if len(required_node_indices) > self.max_nodes:
            raise ValueError("required nodes exceed max_nodes")
        self.required_node_indices = required_node_indices
        length = len(panel.times)
        train_end = int(length * split_ratios[0])
        val_end = int(length * (split_ratios[0] + split_ratios[1]))
        bounds = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, length),
            "all": (0, length),
        }
        lower, upper = bounds[split]
        raw_starts = np.arange(lower, max(lower, upper - seq_len + 1), stride)
        observed_per_time = np.isfinite(panel.values).sum(axis=1, dtype=np.int32)
        cumulative = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(observed_per_time)],
        )
        minimum_observations = min_context_points * min(2, panel.num_nodes)
        totals = cumulative[raw_starts + seq_len] - cumulative[raw_starts]
        eligible = totals >= minimum_observations
        if mask_mode == "natural":
            coverage_events = np.zeros(length + 1, dtype=np.int32)
            np.add.at(coverage_events, panel.coverage_bounds[:, 0], 1)
            np.add.at(coverage_events, panel.coverage_bounds[:, 1] + 1, -1)
            coverage_per_time = np.cumsum(coverage_events[:-1])
            missing_per_time = np.maximum(coverage_per_time - observed_per_time, 0)
            missing_cumulative = np.concatenate(
                [np.zeros(1, dtype=np.int64), np.cumsum(missing_per_time)],
            )
            missing_totals = (
                missing_cumulative[raw_starts + seq_len]
                - missing_cumulative[raw_starts]
            )
            eligible &= missing_totals > 0
        self.starts = raw_starts[eligible].astype(np.int64)
        if not self.starts.size and mask_mode != "natural":
            raise ValueError(
                f"No eligible {split} windows for {panel.variable}; "
                "reduce seq_len/min_context_points",
            )

    def __len__(self) -> int:
        return int(self.starts.size)

    def set_epoch(self, epoch: int) -> None:
        """Vary training masks and node subsets reproducibly between epochs."""
        self.epoch = int(epoch)

    def _rng(self, index: int) -> np.random.Generator:
        epoch = self.epoch if self.split == "train" else 0
        return np.random.default_rng(self.seed + 1_000_003 * epoch + index)

    def _select_nodes(
        self,
        observed: np.ndarray,
        rng: np.random.Generator,
        natural_missing: np.ndarray | None = None,
        required_node_indices: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        counts = observed.sum(axis=0)
        active = np.flatnonzero(counts >= self.min_context_points)
        if active.size == 0:
            active = np.flatnonzero(counts >= 2)
        if active.size == 0:
            active = np.array([int(np.argmax(counts))])
        if natural_missing is not None:
            missing_counts = natural_missing.sum(axis=0)
            needs_imputation = np.flatnonzero((missing_counts > 0) & (counts >= 2))
            ordered_missing = needs_imputation[
                np.lexsort((needs_imputation, -missing_counts[needs_imputation]))
            ]
            active = np.concatenate(
                [ordered_missing, active[~np.isin(active, ordered_missing)]],
            )
        required_nodes = (
            self.required_node_indices
            if required_node_indices is None
            else required_node_indices
        )
        if required_nodes:
            required = np.asarray(required_nodes, dtype=np.int64)
            active = np.concatenate([required, active[~np.isin(active, required)]])
        if active.size > self.max_nodes:
            if self.split == "train" and natural_missing is None:
                active = np.sort(rng.choice(active, self.max_nodes, replace=False))
            elif natural_missing is not None:
                active = active[: self.max_nodes]
            else:
                order = np.lexsort((active, -counts[active]))
                active = np.sort(active[order[: self.max_nodes]])
        return active

    def _point_mask(
        self,
        observed: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return point_mask(observed, rng, float(rng.uniform(*self.point_rate)))

    def _block_mask(
        self,
        observed: np.ndarray,
        rng: np.random.Generator,
        *,
        shared_time: bool,
    ) -> np.ndarray:
        length = int(rng.choice(self.block_lengths))
        return block_mask(
            observed,
            rng,
            length,
            shared_time=shared_time,
            terminal=self.mask_mode == "terminal",
        )

    def _synthetic_mask(
        self,
        observed: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mode = self.mask_mode
        if mode == "natural":
            return np.zeros_like(observed)
        if mode == "mixed":
            mode = cast(
                MaskMode,
                str(
                    rng.choice(
                        ("point", "block", "spatial_block"),
                        p=(0.3, 0.5, 0.2),
                    ),
                ),
            )
        if mode == "point":
            mask = self._point_mask(observed, rng)
        else:
            mask = self._block_mask(
                observed,
                rng,
                shared_time=mode in {"spatial_block", "terminal"},
            )
        if not mask.any():
            mask = self._point_mask(observed, rng)
        return mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._window(index, self.required_node_indices)

    def _window(  # noqa: PLR0914
        self, index: int, required_nodes: tuple[int, ...]
    ) -> dict[str, torch.Tensor]:
        start = int(self.starts[index])
        end = start + self.seq_len
        rng = self._rng(index)
        full_values = self.panel.values[start:end]
        full_observed = np.isfinite(full_values)
        full_positions = np.arange(start, end, dtype=np.int64)[:, None]
        all_bounds = self.panel.coverage_bounds
        full_inside_coverage = (full_positions >= all_bounds[None, :, 0]) & (
            full_positions <= all_bounds[None, :, 1]
        )
        full_natural_mask = full_inside_coverage & ~full_observed
        nodes = self._select_nodes(
            full_observed,
            rng,
            full_natural_mask if self.mask_mode == "natural" else None,
            required_nodes,
        )

        values = np.asarray(full_values[:, nodes], dtype=np.float32).copy()
        observed = np.isfinite(values)
        synthetic_mask = self._synthetic_mask(observed, rng)
        positions = np.arange(start, end, dtype=np.int64)[:, None]
        selected_bounds = self.panel.coverage_bounds[nodes]
        inside_coverage = (positions >= selected_bounds[None, :, 0]) & (
            positions <= selected_bounds[None, :, 1]
        )
        natural_mask = inside_coverage & ~observed
        active_mask = natural_mask if self.mask_mode == "natural" else synthetic_mask
        visible_values = values.copy()
        visible_values[synthetic_mask] = np.nan
        baseline = _temporal_baseline(visible_values, self.panel.variable)
        baseline_available = np.isfinite(baseline)

        node_count = len(nodes)
        coordinates = np.zeros((self.max_nodes, 2), dtype=np.float32)
        coordinates[:node_count, 0] = self.panel.coordinates[nodes, 0] / 90.0
        coordinates[:node_count, 1] = self.panel.coordinates[nodes, 1] / 180.0
        node_mask = np.zeros(self.max_nodes, dtype=bool)
        node_mask[:node_count] = True

        center = np.zeros(self.max_nodes, dtype=np.float32)
        scale = np.ones(self.max_nodes, dtype=np.float32)
        direction = self.panel.variable == "wind_direction"
        floor = _SCALE_FLOORS.get(self.panel.variable, 1.0)
        for node in range(node_count):
            visible = visible_values[:, node]
            finite = visible[np.isfinite(visible)]
            if not finite.size:
                continue
            if direction:
                radians = np.deg2rad(finite)
                center[node] = float(
                    np.mod(
                        np.rad2deg(
                            np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
                        ),
                        360.0,
                    ),
                )
                scale[node] = 180.0
            else:
                center[node] = float(np.median(finite))
                q25, q75 = np.percentile(finite, (25, 75))
                robust_scale = float((q75 - q25) / 1.349)
                scale[node] = max(robust_scale, floor)

        padded_values = np.full(
            (self.seq_len, self.max_nodes), np.nan, dtype=np.float32
        )
        padded_visible = np.full_like(padded_values, np.nan)
        padded_baseline = np.full_like(padded_values, np.nan)
        padded_values[:, :node_count] = values
        padded_visible[:, :node_count] = visible_values
        padded_baseline[:, :node_count] = baseline
        padded_mask = np.zeros_like(padded_values, dtype=bool)
        padded_mask[:, :node_count] = active_mask
        padded_observed = np.zeros_like(padded_values, dtype=bool)
        padded_observed[:, :node_count] = np.isfinite(visible_values)
        padded_baseline_available = np.zeros_like(padded_values, dtype=bool)
        padded_baseline_available[:, :node_count] = baseline_available

        expanded_center = center[np.newaxis, :]
        expanded_scale = scale[np.newaxis, :]
        if direction:
            station_norm = np.asarray(
                _circular_difference(padded_visible, expanded_center) / 180.0,
                dtype=np.float32,
            )
            baseline_norm = np.asarray(
                _circular_difference(padded_baseline, expanded_center) / 180.0,
                dtype=np.float32,
            )
            target_norm = np.asarray(
                _circular_difference(padded_values, expanded_center) / 180.0,
                dtype=np.float32,
            )
        else:
            station_norm = np.asarray(
                (padded_visible - expanded_center) / expanded_scale,
                dtype=np.float32,
            )
            baseline_norm = np.asarray(
                (padded_baseline - expanded_center) / expanded_scale,
                dtype=np.float32,
            )
            target_norm = np.asarray(
                (padded_values - expanded_center) / expanded_scale,
                dtype=np.float32,
            )

        aux_values = np.full_like(padded_values, np.nan)
        if self.panel.auxiliary is not None:
            aux_values[:, :node_count] = self.panel.auxiliary[start:end, nodes]
        aux_observed = np.isfinite(aux_values)
        if direction:
            aux_norm = np.asarray(
                _circular_difference(aux_values, expanded_center) / 180.0,
                dtype=np.float32,
            )
        else:
            aux_norm = np.asarray(
                (aux_values - expanded_center) / expanded_scale,
                dtype=np.float32,
            )

        times = self.panel.times[start:end]
        hours = (times.astype("datetime64[h]").astype(np.int64) % 24).astype(float)
        year_start = times.astype("datetime64[Y]")
        day_of_year = (times.astype("datetime64[D]") - year_start).astype(int)
        hour_phase = 2.0 * np.pi * hours / 24.0
        year_phase = 2.0 * np.pi * day_of_year / 365.2425

        features = np.zeros(
            (self.seq_len, self.max_nodes, len(INPUT_FEATURES)),
            dtype=np.float32,
        )
        features[..., 0] = np.nan_to_num(station_norm)
        features[..., 1] = np.nan_to_num(baseline_norm)
        features[..., 2] = np.nan_to_num(aux_norm)
        features[..., 3] = padded_observed
        features[..., 4] = aux_observed
        features[..., 5] = padded_baseline_available
        features[..., 6] = np.sin(hour_phase)[:, None]
        features[..., 7] = np.cos(hour_phase)[:, None]
        features[..., 8] = np.sin(year_phase)[:, None]
        features[..., 9] = np.cos(year_phase)[:, None]
        if direction:
            baseline_radians = np.deg2rad(padded_baseline)
            features[..., 10] = np.nan_to_num(np.sin(baseline_radians))
            features[..., 11] = np.nan_to_num(np.cos(baseline_radians))
        features[:, ~node_mask, :] = 0.0

        return {
            "features": torch.from_numpy(features),
            "target": torch.from_numpy(np.nan_to_num(target_norm).astype(np.float32)),
            "target_physical": torch.from_numpy(
                np.nan_to_num(padded_values).astype(np.float32),
            ),
            "baseline": torch.from_numpy(
                np.nan_to_num(baseline_norm).astype(np.float32),
            ),
            "baseline_physical": torch.from_numpy(
                np.nan_to_num(padded_baseline).astype(np.float32),
            ),
            "evaluation_mask": torch.from_numpy(padded_mask),
            "visible_mask": torch.from_numpy(padded_observed),
            "node_mask": torch.from_numpy(node_mask),
            "coordinates": torch.from_numpy(coordinates),
            "center": torch.from_numpy(center),
            "scale": torch.from_numpy(scale),
            "timestamps": torch.from_numpy(times.astype(np.int64)),
            "node_indices": torch.from_numpy(
                np.pad(nodes, (0, self.max_nodes - node_count), constant_values=-1),
            ),
            "start_index": torch.tensor(start, dtype=torch.int64),
        }


class NaturalGapWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Cover every repairable station with bounded, training-sized node groups.

    Each target is scored once per window. Other nodes supply context only.
    Windows without two local observations for a target leave it to the
    explicit full-history interpolation fallback.
    """

    def __init__(self, windows: SpatiotemporalWindowDataset) -> None:
        if windows.mask_mode != "natural":
            raise ValueError("Natural gap batching requires natural windows")
        self.windows = windows
        self.groups: list[tuple[int, tuple[int, ...]]] = []
        panel = windows.panel
        for index, start in enumerate(windows.starts):
            observed = np.isfinite(panel.values[start : start + windows.seq_len])
            times = np.arange(start, start + windows.seq_len)[:, None]
            inside = (times >= panel.coverage_bounds[None, :, 0]) & (
                times <= panel.coverage_bounds[None, :, 1]
            )
            targets = np.flatnonzero(
                (inside & ~observed).any(axis=0) & (observed.sum(axis=0) >= 2)
            )
            for offset in range(0, len(targets), windows.max_nodes):
                group = tuple(
                    int(node) for node in targets[offset : offset + windows.max_nodes]
                )
                self.groups.append((index, group))

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window_index, targets = self.groups[index]
        sample = self.windows._window(window_index, targets)
        node_indices = sample["node_indices"]
        is_target = torch.isin(node_indices, node_indices.new_tensor(targets))
        sample["evaluation_mask"][:, ~is_target] = False
        return sample
