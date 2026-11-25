"""Hourly geometry, visible donor selection and physical seasonal references."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .model import VARIABLES


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sample_fingerprint(sample: dict[str, torch.Tensor]) -> str:
    """Identify actual inputs, targets, station order and masks, not only counts."""
    digest = hashlib.sha256()
    for name in sorted(sample):
        array = sample[name].detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


_NS_PER_HOUR = 3_600_000_000_000


_SCALE_FLOORS = {
    "temperature": 1.0,
    "precipitation": 0.25,
    "humidity": 5.0,
    "pressure": 1.0,
    "wind_speed": 0.5,
    "wind_direction": 180.0,
}


Regime = Literal["other-stations", "external-only", "no-contemporaneous-inputs"]


REGIMES: tuple[Regime, ...] = (
    "other-stations",
    "external-only",
    "no-contemporaneous-inputs",
)


@dataclass(frozen=True)
class LongPanel:
    times: np.ndarray
    station_ids: tuple[str, ...]
    coordinates: np.ndarray
    values: np.ndarray
    auxiliary: np.ndarray
    availability: np.ndarray
    train_end: int
    source_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class Outage:
    start: int
    end: int
    node: int
    regime: Regime

    def validate(self, panel: LongPanel) -> None:
        if (
            self.regime not in REGIMES
            or not 0 <= self.start < self.end <= len(panel.times)
            or not 0 <= self.node < len(panel.station_ids)
            or self.start < panel.train_end < self.end
        ):
            raise ValueError("Invalid global outage or train/validation crossing")

    @property
    def identity(self) -> str:
        return digest(
            {
                "start": self.start,
                "end": self.end,
                "node": self.node,
                "regime": self.regime,
            }
        )


@dataclass(frozen=True)
class HistoryReference:
    center: np.ndarray
    scale: np.ndarray
    monthly: np.ndarray
    counts: np.ndarray
    sha256: str


def mean_direction(values: np.ndarray) -> float:
    angles = np.deg2rad(values.astype(np.float64))
    sine, cosine = np.sin(angles).mean(), np.cos(angles).mean()
    return (
        float(np.rad2deg(np.arctan2(sine, cosine)) % 360)
        if np.hypot(sine, cosine) > 1e-6
        else float(values[0] % 360)
    )


def fit_history(
    panel: LongPanel,
    outage: Outage | None = None,
    *,
    humidity_reference_valid: np.ndarray | None = None,
) -> HistoryReference:
    """Mask the full forbidden TRAIN episode before fitting any reference statistic."""
    if outage is not None:
        outage.validate(panel)
    history = panel.values[: panel.train_end].copy()
    history[~panel.availability[: panel.train_end]] = np.nan
    if humidity_reference_valid is not None:
        mask = humidity_reference_valid
        if (
            mask.shape != panel.values.shape[:2]
            or mask.dtype.kind not in "buif"
            or not np.all(np.isnan(mask) | (mask == 0) | (mask == 1))
        ):
            raise ValueError(
                "humidity_reference_valid requires a binary station-hour mask"
            )
        history[..., VARIABLES.index("humidity")][mask[: panel.train_end] != 1] = np.nan
    if outage is not None and outage.start < panel.train_end:
        if outage.regime == "other-stations":
            history[outage.start : outage.end, outage.node] = np.nan
        else:
            history[outage.start : outage.end] = np.nan
    months = (
        panel.times[: panel.train_end].astype("datetime64[M]").astype(np.int64) % 12
    )
    nodes = len(panel.station_ids)
    center = np.zeros((nodes, 6), dtype=np.float32)
    scale = np.ones_like(center)
    monthly = np.zeros((12, nodes, 6), dtype=np.float32)
    counts = np.isfinite(history).sum(axis=0).astype(np.int64)
    for j, variable in enumerate(VARIABLES):
        whole = history[..., j][np.isfinite(history[..., j])]
        if len(whole) < 4:
            continue  # A new network may measure only a subset of the variables.
        mean = (
            mean_direction if j == 5 else lambda x: float(x.astype(np.float64).mean())
        )
        global_mean = mean(whole)
        global_center = mean_direction(whole) if j == 5 else float(np.median(whole))
        q = np.percentile(whole, [25, 75])
        global_scale = (
            180.0
            if j == 5
            else max(float((q[1] - q[0]) / 1.349), _SCALE_FLOORS[variable])
        )
        global_month = []
        for month in range(12):
            vv = history[months == month, :, j]
            vv = vv[np.isfinite(vv)]
            global_month.append(mean(vv) if len(vv) >= 24 else global_mean)
        for node in range(nodes):
            vv = history[:, node, j]
            finite = vv[np.isfinite(vv)]
            if len(finite) >= 4:
                center[node, j] = (
                    mean_direction(finite) if j == 5 else np.median(finite)
                )
                q = np.percentile(finite, [25, 75])
                scale[node, j] = (
                    180.0
                    if j == 5
                    else max(float((q[1] - q[0]) / 1.349), _SCALE_FLOORS[variable])
                )
                annual = mean(finite)
            else:
                center[node, j] = global_center
                scale[node, j] = global_scale
                annual = global_mean
            for month in range(12):
                vv = history[months == month, node, j]
                vv = vv[np.isfinite(vv)]
                monthly[month, node, j] = (
                    mean(vv)
                    if len(vv) >= 24
                    else annual
                    if len(finite) >= 4
                    else global_month[month]
                )
    fingerprint = sample_fingerprint(
        {
            "center": torch.from_numpy(center),
            "scale": torch.from_numpy(scale),
            "monthly": torch.from_numpy(monthly),
            "counts": torch.from_numpy(counts),
        }
    )
    return HistoryReference(center, scale, monthly, counts, fingerprint)


def nearest_nodes(panel: LongPanel, node: int, count: int = 24) -> np.ndarray:
    coordinates = panel.coordinates.astype(np.float64)
    lon_delta = np.abs(coordinates[:, 1] - coordinates[node, 1])
    lon_delta = np.minimum(lon_delta, 360.0 - lon_delta)
    distance = np.square(coordinates[:, 0] - coordinates[node, 0]) + np.square(
        lon_delta * np.cos(np.deg2rad(coordinates[node, 0]))
    )
    others = [n for n in np.argsort(distance, kind="stable") if n != node]
    return np.array([node, *others[: count - 1]], dtype=np.int64)


def prepare_outage_span(
    panel: LongPanel,
    outage: Outage,
    reference: HistoryReference,
    *,
    max_nodes: int = 24,
    hidden_hours: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """One globally masked span shared by all context lengths and output cores."""
    outage.validate(panel)
    left = outage.start - 48
    right = outage.start + int(np.ceil((outage.end - outage.start) / 72)) * 72 + 48
    if left < 0 or right > len(panel.times):
        raise ValueError(
            "Outage context must stay within its declared train/validation split"
        )
    nodes = nearest_nodes(panel, outage.node, max_nodes)
    physical = panel.values[left:right].copy()
    available = panel.availability[left:right]
    visible = np.isfinite(physical) & available
    auxiliary = panel.auxiliary[left:right].copy()
    aux_visible = np.isfinite(auxiliary) & available
    gap = slice(outage.start - left, outage.end - left)
    if outage.regime == "other-stations":
        if hidden_hours is None:
            visible[gap, outage.node] = False
        else:
            if hidden_hours.dtype != bool or hidden_hours.shape != (
                outage.end - outage.start,
            ):
                raise ValueError("Expected one boolean per hour in the station outage")
            visible[gap, outage.node] &= ~hidden_hours[:, None]
    else:
        if hidden_hours is not None:
            raise ValueError("Hourly target masks require the other-stations regime")
        visible[gap] = False
    if outage.regime == "no-contemporaneous-inputs":
        aux_visible[gap] = False
    allowed = np.where(visible, physical, np.nan)
    aux = np.where(aux_visible, auxiliary, np.nan)
    months = panel.times[left:right].astype("datetime64[M]").astype(np.int64) % 12
    count = len(nodes)
    baseline = reference.monthly[months][:, nodes].copy()
    source = np.full(baseline.shape, 3, dtype=np.uint8)
    donor_count = np.zeros_like(source)
    structural_variables: np.ndarray = np.asarray(
        panel.availability.any(axis=0), dtype=bool
    )
    for local, node in enumerate(nodes):
        ordered = nearest_nodes(panel, int(node), len(panel.station_ids))[1:]
        for j in range(6):
            donors = ordered[structural_variables[ordered, j]][:8]
            vv = allowed[:, donors, j]
            nn = np.isfinite(vv).sum(axis=1)
            donor_count[:, local, j] = nn
            external = np.isfinite(aux[:, node, j])
            baseline[:, local, j] = np.where(
                external, aux[:, node, j], baseline[:, local, j]
            )
            source[:, local, j] = np.where(external, 2, source[:, local, j])
            if j == 5:
                angles = np.deg2rad(vv.astype(np.float64))
                mean = (
                    np.rad2deg(
                        np.arctan2(
                            np.nansum(np.sin(angles), axis=1),
                            np.nansum(np.cos(angles), axis=1),
                        )
                    )
                    % 360
                )
            else:
                mean = np.nansum(vv, axis=1) / np.maximum(nn, 1)
            baseline[:, local, j] = np.where(nn >= 3, mean, baseline[:, local, j])
            source[:, local, j] = np.where(nn >= 3, 1, source[:, local, j])
            baseline[:, local, j] = np.where(
                visible[:, node, j], physical[:, node, j], baseline[:, local, j]
            )
            source[:, local, j] = np.where(visible[:, node, j], 4, source[:, local, j])
    selected_available = available[:, nodes]
    center, scale = reference.center[nodes], reference.scale[nodes]

    def normalized(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        clean = np.where(mask, x, center[None])
        result = (clean - center[None]) / scale[None]
        result[..., 5] = ((clean[..., 5] - center[None, :, 5] + 180) % 360 - 180) / 180
        return np.where(mask, result, 0.0).astype(np.float32)

    def pad(x: np.ndarray) -> torch.Tensor:
        result = np.zeros((len(x), max_nodes, 6), dtype=x.dtype)
        result[:, :count] = x
        return torch.from_numpy(result)

    static_center = np.zeros((max_nodes, 6), np.float32)
    static_scale = np.ones_like(static_center)
    static_center[:count] = center
    static_scale[:count] = scale
    static_scale[:, 5] = 180.0
    coords = np.zeros((max_nodes, 2), np.float32)
    coords[:count] = panel.coordinates[nodes] / np.array([90.0, 180.0])
    indices = np.full(max_nodes, -1, dtype=np.int64)
    indices[:count] = nodes
    ref_valid = np.zeros((max_nodes, 6), dtype=bool)
    ref_valid[:count] = reference.counts[nodes] >= 4
    target = panel.values[left:right, outage.node].copy()
    target_mask = np.isfinite(target) & available[:, outage.node]
    target_mask[: outage.start - left] = False
    target_mask[outage.end - left :] = False
    if hidden_hours is not None:
        target_mask[gap] &= hidden_hours[:, None]
    target_physical = np.where(target_mask, target, 0.0).astype(np.float32)
    target_norm = (target - center[0]) / scale[0]
    target_norm[:, 5] = ((target[:, 5] - center[0, 5] + 180) % 360 - 180) / 180
    return {
        "values": pad(normalized(physical[:, nodes], visible[:, nodes])),
        "input_mask": pad(visible[:, nodes]),
        "availability": pad(selected_available),
        "auxiliary": pad(normalized(aux[:, nodes], aux_visible[:, nodes])),
        "auxiliary_mask": pad(aux_visible[:, nodes]),
        "baseline": pad(normalized(baseline, selected_available)),
        "baseline_source": pad(
            np.where(selected_available, source, 0).astype(np.uint8)
        ),
        "donor_count": pad(
            np.where(selected_available, donor_count, 0).astype(np.uint8)
        ),
        "center": torch.from_numpy(static_center),
        "scale": torch.from_numpy(static_scale),
        "reference_valid": torch.from_numpy(ref_valid),
        "coordinates": torch.from_numpy(coords),
        "node_mask": torch.from_numpy(np.arange(max_nodes) < count),
        "node_indices": torch.from_numpy(indices),
        "timestamps": torch.from_numpy(panel.times[left:right].astype(np.int64)),
        "target": torch.from_numpy(
            np.where(target_mask, target_norm, 0.0).astype(np.float32)
        ),
        "target_physical": torch.from_numpy(target_physical),
        "target_mask": torch.from_numpy(target_mask),
        "span_start": torch.tensor(left),
        "episode_start": torch.tensor(outage.start),
        "episode_end": torch.tensor(outage.end),
        "regime": torch.tensor(REGIMES.index(outage.regime)),
    }


def window_from_span(
    span: dict[str, torch.Tensor], core_start: int, context: int
) -> dict[str, torch.Tensor]:
    if context not in (72, 168) or core_start % 72 != int(span["episode_start"]) % 72:
        raise ValueError("Unsupported long-gap context or output ownership")
    left = core_start - (context - 72) // 2 - int(span["span_start"])
    right = left + context
    if left < 0 or right > len(span["values"]):
        raise ValueError("Long-gap window exceeds its globally masked span")
    static = {
        "center",
        "scale",
        "reference_valid",
        "coordinates",
        "node_mask",
        "node_indices",
        "span_start",
        "episode_start",
        "episode_end",
        "regime",
    }
    result = {
        k: v.clone() if k in static else v[left:right].clone() for k, v in span.items()
    }
    core_offset = (context - 72) // 2
    result["target_mask"][:core_offset] = False
    result["target_mask"][core_offset + 72 :] = False
    for key in ("target", "target_physical"):
        result[key] = torch.where(result["target_mask"], result[key], 0.0)
    times = result["timestamps"].numpy().astype("datetime64[ns]")
    hour = times.astype("datetime64[h]").astype(np.int64) % 24
    doy = (times.astype("datetime64[D]") - times.astype("datetime64[Y]")).astype(
        np.int64
    )
    result["calendar"] = torch.from_numpy(
        np.stack(
            [
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * doy / 365.2425),
                np.cos(2 * np.pi * doy / 365.2425),
            ],
            axis=-1,
        ).astype(np.float32)
    )
    result["core_start"] = torch.tensor(core_start)
    validate_long_sample(result)
    return result


def validate_long_sample(sample: dict[str, torch.Tensor]) -> None:
    values = sample["values"]
    if values.ndim != 3 or values.shape[-1] != 6 or values.shape[0] not in (72, 168):
        raise ValueError("Invalid long-gap window geometry")
    steps, nodes, _ = values.shape
    for key in ("values", "auxiliary", "baseline"):
        if (
            sample[key].shape != values.shape
            or sample[key].dtype != torch.float32
            or not bool(torch.isfinite(sample[key]).all())
        ):
            raise ValueError("Invalid long-gap normalized inputs")
    for key in ("input_mask", "availability", "auxiliary_mask"):
        if sample[key].shape != values.shape or sample[key].dtype != torch.bool:
            raise ValueError("Invalid long-gap masks")
    available = sample["availability"] & sample["node_mask"][None, :, None]
    if bool((sample["input_mask"] & ~available).any()) or bool(
        (sample["auxiliary_mask"] & ~available).any()
    ):
        raise ValueError("Long-gap visible inputs must be available")
    if (
        bool(values[~sample["input_mask"]].ne(0).any())
        or bool(sample["auxiliary"][~sample["auxiliary_mask"]].ne(0).any())
        or bool(sample["baseline"][~available].ne(0).any())
    ):
        raise ValueError("Long-gap unavailable payload must be zero")
    for key in ("center", "scale", "reference_valid"):
        if sample[key].shape != (nodes, 6):
            raise ValueError("Invalid long-gap history reference shape")
    if (
        not bool(torch.isfinite(sample["center"]).all())
        or not bool(torch.isfinite(sample["scale"]).all())
        or bool((sample["scale"] <= 0).any())
    ):
        raise ValueError("Invalid long-gap reference values")
    if sample["calendar"].shape != (steps, 4) or not bool(
        torch.isfinite(sample["calendar"]).all()
    ):
        raise ValueError("Invalid source-grid calendar")
    mask = sample["target_mask"]
    if (
        mask.shape != (steps, 6)
        or mask.dtype != torch.bool
        or bool((mask & (sample["input_mask"][:, 0] | ~available[:, 0])).any())
    ):
        raise ValueError("Long-gap targets must be hidden and available")
    for key in ("target", "target_physical"):
        if (
            sample[key].shape != mask.shape
            or not bool(torch.isfinite(sample[key]).all())
            or bool(sample[key][~mask].ne(0).any())
        ):
            raise ValueError("Invalid long-gap targets")
    decoded = sample["center"][0] + sample["target"] * sample["scale"][0]
    error = decoded - sample["target_physical"]
    error[:, 5] = (error[:, 5] + 180) % 360 - 180
    if not torch.allclose(
        error[mask], torch.zeros_like(error[mask]), atol=2e-4, rtol=0
    ):
        raise ValueError("Long-gap target decoding differs")
    if bool((sample["timestamps"].diff() != _NS_PER_HOUR).any()):
        raise ValueError("Long-gap timestamps are not hourly")
