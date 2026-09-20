from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from giano.evaluation.gapfill.cases import BenchmarkCase
from giano.evaluation.gapfill.reconstruction_example import reconstruction_example
from giano.model.imputeformer import ImputeFormerConfig, MeteorologicalImputeFormer
from giano.model.train_imputeformer import TrainingConfig
from giano.spatiotemporal_dataset import INPUT_FEATURES, SpatiotemporalPanel


def _panel() -> SpatiotemporalPanel:
    length = 240
    nodes = 4
    times = np.datetime64("2020-01-01T00", "ns") + np.arange(length) * np.timedelta64(
        1, "h"
    )
    phase = np.arange(length, dtype=np.float32)[:, None]
    values = np.sin(phase / 5.0) + np.arange(nodes, dtype=np.float32)[None, :]
    return SpatiotemporalPanel(
        variable="temperature",
        times=times,
        station_ids=tuple(f"S{index}" for index in range(nodes)),
        coordinates=np.ones((nodes, 2), dtype=np.float32),
        values=values,
        auxiliary=values + 0.1,
        coverage_bounds=np.tile(np.array([0, length - 1]), (nodes, 1)),
        source_paths=tuple(Path(f"S{index}.nc") for index in range(nodes)),
    )


def test_reconstruction_example_keeps_truth_and_hides_corrupted_values() -> None:
    model_config = ImputeFormerConfig(
        input_dim=len(INPUT_FEATURES),
        input_embedding_dim=8,
        spatial_embedding_dim=8,
        num_heads=2,
        num_layers=1,
        projection_tokens=3,
        feed_forward_dim=16,
        coordinate_frequencies=2,
        dropout=0.0,
    )
    model = MeteorologicalImputeFormer(model_config).eval()
    training_config = TrainingConfig(
        seq_len=24,
        stride=12,
        max_nodes=4,
        batch_size=2,
        block_lengths=(3, 6),
    )
    payload = reconstruction_example(
        model,
        {
            "variable": "temperature",
            "training_config": asdict(training_config),
        },
        _panel(),
        split="val",
        case=BenchmarkCase("block", 6),
        seed=42,
        sample_index=0,
        device=torch.device("cpu"),
    )

    series = payload["series"]
    masked = [index for index, value in enumerate(series["mask"]) if value]
    visible = [index for index, value in enumerate(series["mask"]) if not value]
    assert payload["metrics"]["hidden_points"] == 6
    assert masked
    assert all(series["ground_truth"][index] is not None for index in masked)
    assert all(series["corrupted"][index] is None for index in masked)
    assert all(series["reconstruction"][index] is not None for index in masked)
    assert all(
        series["reconstruction"][index] == series["ground_truth"][index]
        for index in visible
    )
    json.dumps(payload, allow_nan=False)


def test_random_reconstruction_selection_is_reproducible() -> None:
    model_config = ImputeFormerConfig(
        input_dim=len(INPUT_FEATURES),
        input_embedding_dim=8,
        spatial_embedding_dim=8,
        num_heads=2,
        num_layers=1,
        projection_tokens=3,
        feed_forward_dim=16,
        coordinate_frequencies=2,
        dropout=0.0,
    )
    model = MeteorologicalImputeFormer(model_config).eval()
    metadata = {
        "variable": "temperature",
        "training_config": asdict(
            TrainingConfig(
                seq_len=24,
                stride=12,
                max_nodes=4,
                batch_size=2,
                block_lengths=(3, 6),
            )
        ),
    }
    first = reconstruction_example(
        model,
        metadata,
        _panel(),
        split="val",
        case=BenchmarkCase("block", 6),
        seed=42,
        sample_index=None,
        random_seed=123,
        device=torch.device("cpu"),
    )
    second = reconstruction_example(
        model,
        metadata,
        _panel(),
        split="val",
        case=BenchmarkCase("block", 6),
        seed=42,
        sample_index=None,
        random_seed=123,
        device=torch.device("cpu"),
    )

    assert first["sample_index"] == second["sample_index"]
    assert first["start_index"] == second["start_index"]
