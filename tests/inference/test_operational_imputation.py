"""Explicit inference methods preserve observations and record their weights."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from giano.baselines.bilstm import BiLSTMConfig, BiLSTMVariant, MeteorologicalBiLSTM
from giano.baselines.train_bilstm import BiLSTMTrainingConfig
from giano.impute import impute_variable, main, parse_args
from giano.model.imputeformer import ImputeFormerConfig, MeteorologicalImputeFormer
from giano.model.train_imputeformer import TrainingConfig
from giano.provenance import file_sha256


def _station(root: Path) -> np.ndarray:
    values = 20 + np.sin(np.arange(120) / 8)
    values[25:31] = np.nan
    directory = root / "train"
    directory.mkdir(parents=True)
    xr.Dataset(
        {"value": ("time", values), "quality": ("time", np.zeros(120))},
        coords={
            "time": np.datetime64("2020-01-01T00", "ns")
            + np.arange(120) * np.timedelta64(1, "h")
        },
        attrs={"station_latitude": 46.1, "station_longitude": 11.1},
    ).to_netcdf(directory / "S0_temperature_merged.nc", engine="h5netcdf")
    return values


def _fair_checkpoint(path: Path, *, variant: BiLSTMVariant = "fair") -> None:
    config = BiLSTMConfig(variant=variant, hidden_size=4)
    model = MeteorologicalBiLSTM(config)
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
    torch.save(
        {
            "model_class": "MeteorologicalBiLSTM",
            "variable": "temperature",
            "variant": variant,
            "model_config": asdict(config),
            "state_dict": model.state_dict(),
            "training_config": asdict(BiLSTMTrainingConfig(seq_len=24, max_nodes=2)),
        },
        path,
    )


def _giano_checkpoint(path: Path) -> None:
    config = ImputeFormerConfig(
        input_embedding_dim=4,
        spatial_embedding_dim=4,
        num_heads=2,
        num_layers=1,
        projection_tokens=2,
        feed_forward_dim=8,
    )
    torch.save(
        {
            "model_class": "MeteorologicalImputeFormer",
            "variable": "temperature",
            "model_config": asdict(config),
            "state_dict": MeteorologicalImputeFormer(config).state_dict(),
            "training_config": asdict(TrainingConfig(seq_len=24, max_nodes=2)),
        },
        path,
    )


@pytest.mark.parametrize("family", ["interpolation", "bilstm_fair"])
def test_explicit_inference_preserves_observations_and_labels_the_actual_method(
    tmp_path, family
):
    values = _station(tmp_path / "data")
    checkpoint = None
    if family == "bilstm_fair":
        checkpoint = tmp_path / "fair.pt"
        _fair_checkpoint(checkpoint)
    outputs = impute_variable(
        tmp_path / "data",
        checkpoint,
        tmp_path / "output",
        model_family=family,
        variable="temperature",
        device=torch.device("cpu"),
    )
    assert len(outputs) == 1
    with xr.open_dataset(outputs[0], engine="h5netcdf") as dataset:
        observed = np.isfinite(values)
        np.testing.assert_array_equal(
            dataset["imputed_value"].values[observed], values[observed]
        )
        np.testing.assert_array_equal(dataset["value"].values, values)
        assert np.isfinite(dataset["imputed_value"].values[~observed]).all()
        assert np.all(dataset["imputation_method"].values[observed] == 0)
        assert np.all(
            dataset["imputation_method"].values[~observed]
            == (3 if family == "bilstm_fair" else 1)
        )
    manifest = json.loads((tmp_path / "output/temperature_manifest.json").read_text())
    assert manifest["model_family"] == family
    assert manifest["checkpoint_sha256"] == (
        file_sha256(checkpoint) if checkpoint else None
    )


def test_inference_rejects_wrong_variable_and_legacy_family(tmp_path):
    path = tmp_path / "fair.pt"
    _fair_checkpoint(path)
    with pytest.raises(ValueError, match="does not match"):
        impute_variable(
            tmp_path,
            path,
            tmp_path / "out",
            model_family="bilstm_fair",
            variable="humidity",
        )
    _fair_checkpoint(path, variant="legacy_retrained")
    with pytest.raises(ValueError, match="only the fair"):
        impute_variable(tmp_path, path, tmp_path / "out", model_family="bilstm_fair")


@pytest.mark.parametrize("family", ["imputeformer", "bilstm_fair", "interpolation"])
def test_cli_uses_explicit_method_data_and_seeded_weights(tmp_path, family):
    _station(tmp_path / "data")
    arguments = [
        "--data-dir",
        str(tmp_path / "data"),
        "--output-dir",
        str(tmp_path / "out"),
        "--variables",
        "temperature",
        "--model-family",
        family,
        "--fallback",
        "none",
    ]
    checkpoint = None
    if family != "interpolation":
        checkpoint = tmp_path / "weights/seed-42/temperature.pt"
        checkpoint.parent.mkdir(parents=True)
        if family == "imputeformer":
            _giano_checkpoint(checkpoint)
        else:
            _fair_checkpoint(checkpoint)
        arguments += ["--checkpoint-dir", str(tmp_path / "weights"), "--seed", "42"]
    assert main(arguments) == 0
    output = tmp_path / "out/temperature"
    assert (output / "S0_temperature_merged_imputed.nc").is_file()
    manifest = json.loads((output / "temperature_manifest.json").read_text())
    assert manifest["model_family"] == family
    assert manifest["checkpoint_sha256"] == (
        file_sha256(checkpoint) if checkpoint else None
    )
    assert manifest["fallback"] == "none"
    assert "routing_policy" not in manifest


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ([], "--data-dir"),
        (["--data-dir", "data"], "--checkpoint-dir"),
        (
            [
                "--data-dir",
                "data",
                "--model-family",
                "interpolation",
                "--checkpoint-dir",
                "weights",
            ],
            "does not use checkpoints",
        ),
        (
            ["--data-dir", "data", "--model-family", "interpolation", "--seed", "42"],
            "does not use checkpoints",
        ),
        (
            ["--data-dir", "data", "--checkpoint-dir", "weights", "--seed", "-1"],
            "seed must be",
        ),
        (
            ["--data-dir", "data", "--checkpoint-dir", "weights", "--seed", str(2**32)],
            "seed must be",
        ),
        (
            [
                "--data-dir",
                "data",
                "--checkpoint-dir",
                "weights",
                "--policy",
                "policy.json",
            ],
            "unrecognized arguments",
        ),
    ],
)
def test_cli_rejects_missing_paths_and_irrelevant_options(arguments, message, capsys):
    with pytest.raises(SystemExit) as error:
        parse_args(arguments)
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("fallback", ["none", "interpolation"])
def test_giano_can_leave_unsupported_natural_gaps_empty(tmp_path, fallback):
    values = _station(tmp_path / "data")
    source = tmp_path / "data/train/S0_temperature_merged.nc"
    with xr.open_dataset(source, engine="h5netcdf") as opened:
        dataset = opened.load()
    values[20:100] = np.nan
    dataset["value"].values[:] = values
    dataset.to_netcdf(source, engine="h5netcdf")
    checkpoint = tmp_path / "giano.pt"
    _giano_checkpoint(checkpoint)
    outputs = impute_variable(
        tmp_path / "data",
        checkpoint,
        tmp_path / "out",
        device=torch.device("cpu"),
        fallback=fallback,
    )
    with xr.open_dataset(outputs[0], engine="h5netcdf") as result:
        counts = result["window_prediction_count"].values
        filled = result["imputed_value"].values
        np.testing.assert_array_equal(result["value"].values, values)
        np.testing.assert_array_equal(
            filled[np.isfinite(values)], values[np.isfinite(values)]
        )
        assert counts[21] > 0 and np.isfinite(filled[21])
        assert counts[50] == 0
        assert np.isnan(filled[50]) if fallback == "none" else np.isfinite(filled[50])
        assert np.all(counts[np.isfinite(values)] == 0)
