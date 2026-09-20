from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from giano.baselines import train_bilstm as bilstm_training
from giano.baselines.bilstm import BiLSTMConfig, MeteorologicalBiLSTM
from giano.baselines.train_bilstm import (
    BiLSTMTrainingConfig,
    _training_loss,
    load_bilstm_checkpoint,
    train_bilstm,
)
from giano.evaluation.gapfill.cases import STANDARD_GAPFILL_CASES, dataset_for_case
from giano.model.train_imputeformer import TrainingConfig
from giano.spatiotemporal_dataset import (
    INPUT_FEATURES,
    SpatiotemporalPanel,
)


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.randn(2, 12, 3, len(INPUT_FEATURES))
    features[..., 3:6] = 1.0
    features[:, 4:8, 1, 0] = 0.0
    features[:, 4:8, 1, 3] = 0.0
    coordinates = torch.rand(2, 3, 2)
    node_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    baseline = torch.randn(2, 12, 3)
    return features, coordinates, node_mask, baseline


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


def test_bilstm_variants_follow_distinct_missing_value_contracts() -> None:
    features, coordinates, node_mask, baseline = _batch()
    legacy = MeteorologicalBiLSTM(BiLSTMConfig("legacy_retrained"))
    fair = MeteorologicalBiLSTM(BiLSTMConfig("fair"))

    legacy_inputs = legacy._inputs(features, coordinates)
    fair_inputs = fair._inputs(features, coordinates)
    legacy_prediction, _ = legacy(features, coordinates, node_mask, baseline)
    fair_prediction, _ = fair(features, coordinates, node_mask, baseline)

    assert torch.any(legacy_inputs == -100)
    assert not torch.any(fair_inputs == -100)
    assert torch.isfinite(fair_inputs).all()
    assert legacy_prediction.shape == baseline.shape
    assert fair_prediction.shape == baseline.shape
    assert torch.count_nonzero(fair_prediction[1, :, 2]) == 0


def test_legacy_and_fair_losses_use_the_declared_scopes() -> None:
    prediction = torch.tensor([[[10.0], [2.0]]])
    batch = {
        "target": torch.zeros_like(prediction),
        "evaluation_mask": torch.tensor([[[False], [True]]]),
        "visible_mask": torch.tensor([[[True], [False]]]),
    }

    fair = _training_loss(prediction, batch, "fair", direction=False)
    legacy = _training_loss(prediction, batch, "legacy_retrained", direction=False)

    assert fair == 2.0
    assert legacy == 52.0


@pytest.mark.parametrize("case", STANDARD_GAPFILL_CASES)
def test_common_case_materializes_identical_masks_for_bilstm_and_giano(case) -> None:
    panel = _panel()
    panel.values[30:36, 0] = np.nan
    giano = dataset_for_case(
        panel,
        "test",
        TrainingConfig(seq_len=32, max_nodes=4, batch_size=2),
        case,
        19,
    )
    bilstm = dataset_for_case(
        panel,
        "test",
        BiLSTMTrainingConfig(seq_len=32, max_nodes=4, batch_size=2),
        case,
        19,
    )

    assert len(giano) == len(bilstm)
    for index in range(len(giano)):
        assert torch.equal(
            giano[index]["evaluation_mask"], bilstm[index]["evaluation_mask"]
        )
        assert torch.equal(giano[index]["node_indices"], bilstm[index]["node_indices"])


def test_bilstm_checkpoint_loading_is_restricted_and_strict(
    monkeypatch, tmp_path: Path
) -> None:
    config = BiLSTMConfig("fair", hidden_size=8)
    source = MeteorologicalBiLSTM(config)
    calls: dict[str, object] = {}

    def fake_load(*args, **kwargs):
        calls.update(kwargs)
        return {
            "model_class": "MeteorologicalBiLSTM",
            "model_config": asdict(config),
            "state_dict": source.state_dict(),
        }

    monkeypatch.setattr(torch, "load", fake_load)
    loaded, _metadata = load_bilstm_checkpoint(
        tmp_path / "checkpoint.pt", torch.device("cpu")
    )

    assert isinstance(loaded, MeteorologicalBiLSTM)
    assert calls["weights_only"] is True


def test_bilstm_full_epoch_configuration_uses_null_batch_limit() -> None:
    config = BiLSTMTrainingConfig()
    config.validate()
    assert config.max_batches_per_epoch is None
    with pytest.raises(ValueError, match="positive or null"):
        BiLSTMTrainingConfig(max_batches_per_epoch=0).validate()


def test_bilstm_training_resumes_from_last_completed_epoch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "train"
    source_dir.mkdir()
    (source_dir / "A_temperature_merged.nc").write_bytes(b"mock panel source v1")
    monkeypatch.setattr(
        bilstm_training, "build_spatiotemporal_panel", lambda *_: _panel()
    )
    model_config = BiLSTMConfig("fair", hidden_size=8)
    training_config = BiLSTMTrainingConfig(
        seq_len=24,
        stride=12,
        max_nodes=4,
        batch_size=2,
        max_epochs=2,
        max_batches_per_epoch=1,
        max_validation_batches=1,
        patience=3,
        block_lengths=(3, 6),
    )
    checkpoint = tmp_path / "temperature.pt"
    resume_checkpoint = tmp_path / "temperature.resume.pt"
    original_loss = bilstm_training._training_loss
    loss_calls = 0

    def interrupt_during_second_epoch(*args, **kwargs):
        nonlocal loss_calls
        loss_calls += 1
        if loss_calls == 2:
            raise KeyboardInterrupt
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(
        bilstm_training,
        "_training_loss",
        interrupt_during_second_epoch,
    )
    with pytest.raises(KeyboardInterrupt):
        train_bilstm(
            tmp_path,
            "temperature",
            checkpoint,
            model_config,
            training_config,
            torch.device("cpu"),
        )

    progress = torch.load(resume_checkpoint, map_location="cpu", weights_only=True)
    assert progress["checkpoint_kind"] == "training_progress"
    assert progress["training_state"]["epochs_completed"] == 1

    monkeypatch.setattr(bilstm_training, "_training_loss", original_loss)
    train_bilstm(
        tmp_path,
        "temperature",
        checkpoint,
        model_config,
        training_config,
        torch.device("cpu"),
        resume=True,
    )

    completed = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert completed["training_dataset"] == progress["training_dataset"]
    assert completed["training_run"]["epochs_completed"] == 2
    assert completed["training_run"]["resume_count"] == 1
    assert not resume_checkpoint.exists()

    monkeypatch.setattr(
        bilstm_training,
        "build_spatiotemporal_panel",
        lambda *_: pytest.fail("completed run should be skipped before loading data"),
    )
    skipped = train_bilstm(
        tmp_path,
        "temperature",
        checkpoint,
        model_config,
        training_config,
        torch.device("cpu"),
        resume=True,
    )
    assert asdict(skipped) == completed["validation"]
