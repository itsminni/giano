"""Leakage-safe adapters from trained gap fillers to forecasting histories."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from giano.prediction import physical_prediction
from giano.spatiotemporal_dataset import (
    SpatiotemporalPanel,
    SpatiotemporalWindowDataset,
)

CheckpointFamily = Literal["imputeformer", "bilstm"]


def _training_dimension(metadata: Mapping[str, Any], name: str) -> int:
    raw = metadata.get("training_config")
    if not isinstance(raw, dict) or not isinstance(raw.get(name), int):
        raise ValueError(f"checkpoint lacks integer training_config.{name}")
    return int(raw[name])


class CheckpointHistoryRepairer:
    """Adapt one project checkpoint to a single rolling-origin station history."""

    def __init__(
        self,
        model: torch.nn.Module,
        metadata: Mapping[str, Any],
        panel: SpatiotemporalPanel,
        station: str,
        device: torch.device,
    ) -> None:
        if metadata.get("variable") != panel.variable:
            raise ValueError("checkpoint and panel variables do not match")
        try:
            target_node = panel.station_ids.index(station)
        except ValueError as error:
            raise ValueError(f"station {station} is absent from the panel") from error
        self.model = model.to(device).eval()
        self.metadata = dict(metadata)
        self.panel = panel
        self.station = station
        self.target_node = target_node
        self.device = device
        self.seq_len = _training_dimension(metadata, "seq_len")
        self.max_nodes = _training_dimension(metadata, "max_nodes")
        self.min_context_points = _training_dimension(
            metadata,
            "min_context_points",
        )

    def _history_panel(
        self,
        corrupted: np.ndarray,
        visible: np.ndarray,
        history_times: np.ndarray,
    ) -> SpatiotemporalPanel:
        times = np.asarray(history_times).astype("datetime64[ns]")
        positions = np.searchsorted(self.panel.times, times)
        if (
            len(times) != self.seq_len
            or positions[-1] >= len(self.panel.times)
            or not np.array_equal(self.panel.times[positions], times)
            or not np.array_equal(
                positions, np.arange(positions[0], positions[0] + len(times))
            )
        ):
            raise ValueError("repair history must be one aligned checkpoint window")
        values = self.panel.values[positions].copy()
        values[:, self.target_node] = np.asarray(corrupted, dtype=np.float32)
        values[~visible, self.target_node] = np.nan
        auxiliary = (
            None
            if self.panel.auxiliary is None
            else self.panel.auxiliary[positions].copy()
        )
        start = int(positions[0])
        end = int(positions[-1])
        bounds = np.zeros_like(self.panel.coverage_bounds)
        for node, (lower, upper) in enumerate(self.panel.coverage_bounds):
            overlap_lower = max(start, int(lower))
            overlap_upper = min(end, int(upper))
            bounds[node] = (
                (overlap_lower - start, overlap_upper - start)
                if overlap_lower <= overlap_upper
                else (0, -1)
            )
        return SpatiotemporalPanel(
            variable=self.panel.variable,
            times=times,
            station_ids=self.panel.station_ids,
            coordinates=self.panel.coordinates,
            values=values,
            auxiliary=auxiliary,
            coverage_bounds=bounds,
            source_paths=self.panel.source_paths,
        )

    @torch.no_grad()
    def __call__(
        self,
        corrupted: np.ndarray,
        visible: np.ndarray,
        history_times: np.ndarray,
    ) -> np.ndarray:
        """Repair hidden target values using no timestamps after the origin."""
        corrupted_values = np.asarray(corrupted, dtype=np.float64)
        visible_values = np.asarray(visible, dtype=bool)
        if corrupted_values.shape != (self.seq_len,) or visible_values.shape != (
            self.seq_len,
        ):
            raise ValueError("checkpoint repairer received the wrong history shape")
        if visible_values.sum() < 2 or visible_values.all():
            raise ValueError("checkpoint repairer requires visible and hidden values")
        panel = self._history_panel(
            corrupted_values,
            visible_values,
            history_times,
        )
        dataset = SpatiotemporalWindowDataset(
            panel,
            "all",
            seq_len=self.seq_len,
            stride=self.seq_len,
            max_nodes=self.max_nodes,
            min_context_points=self.min_context_points,
            seed=0,
            mask_mode="natural",
            block_lengths=(min(3, self.seq_len - 1),),
            required_node_indices=(self.target_node,),
        )
        cpu_batch = dataset[0]
        node_indices = cpu_batch["node_indices"].numpy()
        target_positions = np.flatnonzero(node_indices == self.target_node)
        if target_positions.size != 1:
            raise RuntimeError("target station was not selected for imputation")
        batch = {
            key: value.unsqueeze(0).to(self.device) for key, value in cpu_batch.items()
        }
        prediction_norm, _ = self.model(
            batch["features"],
            batch["coordinates"],
            batch["node_mask"],
            batch["baseline"],
        )
        prediction = physical_prediction(
            prediction_norm,
            batch["center"],
            batch["scale"],
            variable=self.panel.variable,
        )[0, :, int(target_positions[0])]
        repaired = corrupted_values.copy()
        repaired[~visible_values] = prediction.detach().cpu().numpy()[~visible_values]
        repaired[visible_values] = corrupted_values[visible_values]
        return repaired


def checkpoint_history_repairer(
    checkpoint: Path,
    panel: SpatiotemporalPanel,
    station: str,
    *,
    family: CheckpointFamily,
    device: torch.device,
) -> CheckpointHistoryRepairer:
    """Load a restricted project checkpoint and expose a history repairer."""
    model: torch.nn.Module
    metadata: dict[str, Any]
    if family == "imputeformer":
        from giano.model.train_imputeformer import load_imputeformer_checkpoint

        model, metadata = load_imputeformer_checkpoint(checkpoint, device)
    else:
        from giano.baselines.train_bilstm import load_bilstm_checkpoint

        model, metadata = load_bilstm_checkpoint(checkpoint, device)
    return CheckpointHistoryRepairer(model, metadata, panel, station, device)
