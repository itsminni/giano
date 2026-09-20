from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from giano.data_processing.unify_dataset import process_csv_only_pairs
from giano.impute import impute_variable
from giano.model import train_imputeformer as imputeformer_training
from giano.model.imputeformer import (
    ImputeFormerConfig,
    MeteorologicalImputeFormer,
    temporal_spectral_regularization,
)
from giano.model.train_imputeformer import (
    TrainingConfig,
    load_imputeformer_checkpoint,
    train_variable,
)
from giano.spatiotemporal_dataset import (
    INPUT_FEATURES,
    SpatiotemporalWindowDataset,
    _temporal_baseline,
    build_spatiotemporal_panel,
)


@pytest.mark.parametrize("block_lengths", [None, [3, 12], (6, 24)])
def test_checkpoint_config_normalizes_without_changing_metadata(block_lengths) -> None:
    values = {"seed": 44, "seq_len": 48, "context_weight": 0.0}
    if block_lengths is not None:
        values["block_lengths"] = block_lengths
    original = dict(values)
    config = TrainingConfig.from_checkpoint({"training_config": values})

    assert config.seed == 44
    assert config.seq_len == 48
    assert config.block_lengths == (
        (3, 6, 12, 24) if block_lengths is None else tuple(block_lengths)
    )
    assert values == original


@pytest.mark.parametrize(
    "metadata", [{}, {"training_config": []}, {"training_config": {"seq_len": 0}}]
)
def test_checkpoint_config_rejects_missing_or_invalid_settings(metadata) -> None:
    with pytest.raises(ValueError):
        TrainingConfig.from_checkpoint(metadata)


def test_checkpoint_config_rejects_unknown_settings() -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        TrainingConfig.from_checkpoint({"training_config": {"unknown_setting": 1}})


def _write_station(
    path: Path,
    station_offset: float,
    *,
    auxiliary: bool,
    quality_value: float = 0.0,
) -> None:
    times = pd.date_range("2020-01-01", periods=240, freq="h")
    phase = np.arange(len(times), dtype=float)
    values = station_offset + np.sin(phase / 5.0)
    values[30:33] = np.nan
    data_vars: dict[str, tuple[str, np.ndarray]] = {
        "value": ("time", values),
        "quality": ("time", np.full(len(times), quality_value)),
    }
    if auxiliary:
        data_vars["temperature"] = ("time", values + 0.2)
    dataset = xr.Dataset(
        data_vars,
        coords={"time": times},
        attrs={
            "station_latitude": 46.0 + station_offset / 10,
            "station_longitude": 11.0 + station_offset / 10,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path, engine="h5netcdf")


def test_panel_and_custom_masks_preserve_available_sources(tmp_path: Path) -> None:
    for index, split in enumerate(("train", "val", "test")):
        _write_station(
            tmp_path / split / f"S{index}_temperature_merged.nc",
            float(index),
            auxiliary=index != 1,
        )
    panel = build_spatiotemporal_panel(tmp_path, "temperature")

    assert panel.values.shape == (240, 3)
    assert panel.auxiliary is not None
    missing_aux_node = panel.station_ids.index("S1")
    assert np.isnan(panel.auxiliary[:, missing_aux_node]).all()

    for mask_mode in ("point", "block", "spatial_block", "mixed"):
        windowed = SpatiotemporalWindowDataset(
            panel,
            "train",
            seq_len=24,
            stride=12,
            max_nodes=3,
            seed=7,
            mask_mode=mask_mode,
            block_lengths=(3, 6),
        )
        sample = windowed[0]
        assert sample["features"].shape == (24, 3, len(INPUT_FEATURES))
        assert sample["evaluation_mask"].any()
        assert torch.isfinite(sample["features"]).all()
        assert not torch.any(sample["features"] == -100)


def test_missing_auxiliary_is_an_explicit_zero_flag(tmp_path: Path) -> None:
    for index, split in enumerate(("train", "val", "test")):
        path = tmp_path / split / f"S{index}_humidity_merged.nc"
        _write_station(path, float(index), auxiliary=False)
    panel = build_spatiotemporal_panel(tmp_path, "humidity")
    sample = SpatiotemporalWindowDataset(
        panel,
        "train",
        seq_len=24,
        max_nodes=3,
        mask_mode="point",
    )[0]

    assert panel.auxiliary is None
    assert torch.count_nonzero(sample["features"][..., 2]) == 0
    assert torch.count_nonzero(sample["features"][..., 4]) == 0


def test_station_only_preprocessing_keeps_coordinates_without_auxiliary(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "raw" / "T0001_humidity.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=8, freq="15min"),
            "value": np.linspace(40.0, 47.0, 8),
            "quality": np.ones(8),
        },
    ).to_csv(csv_path, index=False)

    outputs = process_csv_only_pairs(
        {("T0001", "humidity"): str(csv_path)},
        {},
        [],
        tmp_path / "merged",
        {"T0001": (46.1, 11.2)},
    )

    assert len(outputs) == 1
    with xr.open_dataset(outputs[0], engine="h5netcdf") as processed:
        assert set(processed.data_vars) == {
            "value",
            "quality",
            "raw_quality_max",
            "sample_count",
            "valid_sample_count",
            "valid_sample_fraction",
        }
        assert processed["value"].attrs["units"] == "%"
        assert processed.attrs["preprocessing_version"] == "raw-filter-before-hourly-v2"
        assert processed.attrs["station_latitude"] == 46.1
        assert processed.attrs["station_longitude"] == 11.2


def test_wind_auxiliary_components_must_align_with_station_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train" / "S0_wind_speed_merged.nc"
    path.parent.mkdir(parents=True)
    dataset = xr.Dataset(
        {
            "value": ("time", np.ones(8)),
            "wind_u": ("aux_time", np.ones(8)),
            "wind_v": ("aux_time", np.ones(8)),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=8, freq="h"),
            "aux_time": pd.date_range("2020-01-01", periods=8, freq="h"),
        },
        attrs={"station_latitude": 46.1, "station_longitude": 11.2},
    )
    dataset.to_netcdf(path, engine="h5netcdf")

    with pytest.raises(ValueError, match="wind components are not aligned"):
        build_spatiotemporal_panel(tmp_path, "wind_speed")


def test_processed_panel_does_not_reinterpret_provider_quality(tmp_path: Path) -> None:
    for index, split in enumerate(("train", "val", "test")):
        _write_station(
            tmp_path / split / f"S{index}_temperature_merged.nc",
            float(index),
            auxiliary=True,
            quality_value=151.0,
        )
    panel = build_spatiotemporal_panel(tmp_path, "temperature")

    assert np.isfinite(panel.values[0]).all()


def test_zero_initialized_imputeformer_starts_from_interpolation() -> None:
    config = ImputeFormerConfig(
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
    model = MeteorologicalImputeFormer(config)
    features = torch.randn(2, 12, 4, len(INPUT_FEATURES))
    coordinates = torch.rand(2, 4, 2)
    node_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    baseline = torch.randn(2, 12, 4)

    prediction, residual = model(features, coordinates, node_mask, baseline)

    assert torch.count_nonzero(residual) == 0
    assert torch.equal(prediction, baseline)
    regularization = temporal_spectral_regularization(
        prediction,
        torch.zeros_like(prediction),
        visible_mask=torch.zeros_like(prediction, dtype=torch.bool),
        node_mask=node_mask,
    )
    assert torch.isfinite(regularization)


def test_wind_direction_baseline_crosses_zero_circularly() -> None:
    values = np.array([[350.0], [np.nan], [10.0]], dtype=np.float32)
    baseline = _temporal_baseline(values, "wind_direction")

    assert baseline[1, 0] == 0.0


def test_nonzero_readout_still_preserves_observations_short_gaps_and_padding() -> None:
    model = MeteorologicalImputeFormer(ImputeFormerConfig(dropout=0.0)).eval()
    final_layer = model.readout[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    with torch.no_grad():
        final_layer.bias.fill_(0.25)
    features = torch.zeros(1, 12, 3, len(INPUT_FEATURES))
    features[..., 3] = 1.0
    features[:, 2:5, 0, 3] = 0.0
    features[:, 2:8, 1, 3] = 0.0
    features[:, :, 2, 3] = 0.0
    baseline = torch.randn(1, 12, 3)
    node_mask = torch.tensor([[True, True, False]])
    prediction, residual = model(features, torch.zeros(1, 3, 2), node_mask, baseline)
    assert torch.count_nonzero(residual[features[..., 3].bool()]) == 0
    torch.testing.assert_close(prediction[:, :, 0], baseline[:, :, 0], rtol=0, atol=0)
    assert torch.count_nonzero(residual[:, :, 2]) == 0
    assert torch.all(residual[:, 2:8, 1] > 0)
    assert torch.all(residual.abs() <= model.config.max_residual)


def test_checkpoint_loading_is_restricted_and_strict(
    monkeypatch, tmp_path: Path
) -> None:
    config = ImputeFormerConfig(
        input_dim=len(INPUT_FEATURES),
        input_embedding_dim=8,
        spatial_embedding_dim=8,
        num_heads=2,
        num_layers=1,
        projection_tokens=3,
        feed_forward_dim=16,
        coordinate_frequencies=2,
    )
    source = MeteorologicalImputeFormer(config)
    calls: dict[str, object] = {}

    def fake_load(*args, **kwargs):
        calls.update(kwargs)
        return {
            "model_class": "MeteorologicalImputeFormer",
            "model_config": asdict(config),
            "state_dict": source.state_dict(),
        }

    monkeypatch.setattr(torch, "load", fake_load)
    loaded, _ = load_imputeformer_checkpoint(
        tmp_path / "checkpoint.pt",
        torch.device("cpu"),
    )

    assert isinstance(loaded, MeteorologicalImputeFormer)
    assert calls["weights_only"] is True


def test_full_epoch_training_records_history_and_steps(tmp_path: Path) -> None:
    data_root = tmp_path / "processed"
    for index, split in enumerate(("train", "val", "test")):
        _write_station(
            data_root / split / f"S{index}_temperature_merged.nc",
            float(index),
            auxiliary=True,
        )
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
    training_config = TrainingConfig(
        seq_len=24,
        stride=12,
        max_nodes=3,
        batch_size=2,
        max_epochs=1,
        max_batches_per_epoch=None,
        max_validation_batches=1,
        patience=1,
        block_lengths=(3, 6),
    )
    checkpoint = tmp_path / "temperature.pt"

    train_variable(
        data_root,
        "temperature",
        checkpoint,
        model_config,
        training_config,
        torch.device("cpu"),
    )

    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    run = raw["training_run"]
    assert raw["format_version"] == 2
    assert raw["training_objective"] == imputeformer_training.training_objective()
    assert run["epochs_completed"] == 1
    assert run["optimizer_steps"] == run["available_train_batches_per_epoch"]
    assert run["used_all_train_batches_each_epoch"] is True
    assert len(run["history"]) == 1
    assert not checkpoint.with_suffix(".pt.tmp").exists()


def test_training_resumes_from_last_completed_epoch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "processed"
    for index, split in enumerate(("train", "val", "test")):
        _write_station(
            data_root / split / f"S{index}_temperature_merged.nc",
            float(index),
            auxiliary=True,
        )
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
    training_config = TrainingConfig(
        seq_len=24,
        stride=12,
        max_nodes=3,
        batch_size=2,
        max_epochs=2,
        max_batches_per_epoch=1,
        max_validation_batches=1,
        patience=3,
        block_lengths=(3, 6),
    )
    checkpoint = tmp_path / "temperature.pt"
    resume_checkpoint = tmp_path / "temperature.resume.pt"
    original_loss = imputeformer_training._loss
    loss_calls = 0

    def interrupt_during_second_epoch(*args, **kwargs):
        nonlocal loss_calls
        loss_calls += 1
        if loss_calls == 2:
            raise KeyboardInterrupt
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(
        imputeformer_training,
        "_loss",
        interrupt_during_second_epoch,
    )
    with pytest.raises(KeyboardInterrupt):
        train_variable(
            data_root,
            "temperature",
            checkpoint,
            model_config,
            training_config,
            torch.device("cpu"),
        )

    progress = torch.load(resume_checkpoint, map_location="cpu", weights_only=True)
    assert progress["checkpoint_kind"] == "training_progress"
    assert progress["training_objective"] == imputeformer_training.training_objective()
    assert progress["training_state"]["epochs_completed"] == 1
    assert not checkpoint.exists()

    monkeypatch.setattr(imputeformer_training, "_loss", original_loss)
    # Never mix optimizer state / best scores from a different loss recipe.
    for previous_objective in (None, {"version": "legacy-2d-spectrum"}):
        incompatible = dict(progress, training_objective=previous_objective)
        torch.save(incompatible, resume_checkpoint)
        before = resume_checkpoint.read_bytes()
        with pytest.raises(ValueError, match="training objective"):
            train_variable(
                data_root,
                "temperature",
                checkpoint,
                model_config,
                training_config,
                torch.device("cpu"),
                resume=True,
            )
        assert resume_checkpoint.read_bytes() == before
        assert not checkpoint.exists()
    torch.save(progress, resume_checkpoint)
    train_variable(
        data_root,
        "temperature",
        checkpoint,
        model_config,
        training_config,
        torch.device("cpu"),
        resume=True,
    )

    completed = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert completed["training_dataset"] == progress["training_dataset"]
    assert completed["training_objective"] == progress["training_objective"]
    assert completed["training_run"]["epochs_completed"] == 2
    assert completed["training_run"]["resume_count"] == 1
    assert not resume_checkpoint.exists()

    monkeypatch.setattr(
        imputeformer_training,
        "build_spatiotemporal_panel",
        lambda *_: pytest.fail("completed run should be skipped before loading data"),
    )
    skipped = train_variable(
        data_root,
        "temperature",
        checkpoint,
        model_config,
        training_config,
        torch.device("cpu"),
        resume=True,
    )
    assert asdict(skipped) == completed["validation"]


def test_natural_gap_imputation_preserves_original_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "processed"
    for index, split in enumerate(("train", "val", "test")):
        _write_station(
            data_root / split / f"S{index}_temperature_merged.nc",
            float(index),
            auxiliary=True,
        )
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
    model = MeteorologicalImputeFormer(model_config)
    training_config = TrainingConfig(
        seq_len=24,
        stride=12,
        max_nodes=2,
        batch_size=2,
        max_epochs=1,
        max_batches_per_epoch=1,
        max_validation_batches=1,
        patience=1,
        block_lengths=(3, 6),
    )
    checkpoint = tmp_path / "temperature.pt"
    torch.save(
        {
            "format_version": 1,
            "model_class": "MeteorologicalImputeFormer",
            "variable": "temperature",
            "state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "validation": {},
        },
        checkpoint,
    )

    node_counts = []
    original_forward = MeteorologicalImputeFormer.forward

    def checked_forward(self, features, coordinates, node_mask, baseline):
        node_counts.append(features.shape[2])
        return original_forward(self, features, coordinates, node_mask, baseline)

    monkeypatch.setattr(MeteorologicalImputeFormer, "forward", checked_forward)
    outputs = impute_variable(
        data_root,
        checkpoint,
        tmp_path / "imputed",
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert len(outputs) == 3
    assert node_counts and set(node_counts) == {2}
    for output in outputs:
        with xr.open_dataset(output, engine="h5netcdf") as reconstructed:
            original = reconstructed["value"].values
            visible = np.isfinite(original)
            np.testing.assert_array_equal(
                reconstructed["imputed_value"].values[visible], original[visible]
            )
            assert np.isnan(original[30:33]).all()
            assert np.isfinite(reconstructed["imputed_value"].values[30:33]).all()
            assert np.all(reconstructed["imputed_mask"].values[30:33] == 1)
            assert np.all(reconstructed["imputation_method"].values[30:33] == 1)
