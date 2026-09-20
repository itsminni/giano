from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from giano.evaluation.forecasting.checkpoint_repairer import CheckpointHistoryRepairer
from giano.spatiotemporal_dataset import SpatiotemporalPanel


class _InterpolationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        node_mask: torch.Tensor,
        baseline: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del features, coordinates, node_mask
        return baseline + self.anchor * 0.0, torch.zeros_like(baseline)


def _panel(*, perturb_future: bool) -> SpatiotemporalPanel:
    length = 24
    positions = np.arange(length, dtype=np.float32)
    values = np.column_stack((positions, positions * 2.0)).astype(np.float32)
    if perturb_future:
        values[12:] += 10000.0
    times = np.datetime64("2020-01-01T00", "ns") + np.arange(length) * np.timedelta64(
        1, "h"
    )
    return SpatiotemporalPanel(
        variable="temperature",
        times=times,
        station_ids=("A", "B"),
        coordinates=np.array([[46.0, 11.0], [46.2, 11.2]], dtype=np.float32),
        values=values,
        auxiliary=None,
        coverage_bounds=np.array([[0, length - 1], [0, length - 1]]),
        source_paths=(Path("A.nc"), Path("B.nc")),
    )


def test_checkpoint_repairer_preserves_visible_values_and_never_reads_future() -> None:
    metadata = {
        "variable": "temperature",
        "training_config": {
            "seq_len": 12,
            "max_nodes": 2,
            "min_context_points": 4,
        },
    }
    visible = np.ones(12, dtype=bool)
    visible[4:8] = False
    clean = np.arange(12, dtype=np.float64)
    corrupted = clean.copy()
    corrupted[~visible] = np.median(clean[visible])
    outputs = []
    for perturb_future in (False, True):
        panel = _panel(perturb_future=perturb_future)
        repairer = CheckpointHistoryRepairer(
            _InterpolationModel(),
            metadata,
            panel,
            "A",
            torch.device("cpu"),
        )
        outputs.append(repairer(corrupted, visible, panel.times[:12]))

    assert np.array_equal(outputs[0], outputs[1])
    assert np.array_equal(outputs[0][visible], clean[visible])
    assert np.allclose(outputs[0][~visible], clean[~visible])
