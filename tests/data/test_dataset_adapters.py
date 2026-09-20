"""Foreign imports preserve provider semantics and work with existing Giano."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import yaml

from giano.data_processing.adapters.base import DatasetAdapter
from giano.data_processing.adapters.csv import station_key
from giano.data_processing.import_dataset import (
    import_dataset,
    main,
    validate_processed_dataset,
)
from giano.evaluation.gapfill.benchmark_imputeformer import benchmark_checkpoint
from giano.impute import impute_variable
from giano.model.imputeformer import ImputeFormerConfig, MeteorologicalImputeFormer
from giano.model.train_imputeformer import TrainingConfig, train_variable
from giano.provenance import file_sha256
from giano.spatiotemporal_dataset import build_spatiotemporal_panel


def _inputs(tmp_path, *, variables=None, cadence=60, frame=None, timezone="UTC"):
    if frame is None:
        frame = pd.DataFrame(
            {
                "station_id": ["station_with_underscores"] * 120,
                "timestamp": pd.date_range("2024-01-01", periods=120, freq="h").astype(
                    str
                ),
                "reading": 10 + np.sin(np.arange(120) / 8),
            }
        )
    frame.to_csv(tmp_path / "observations.csv", index=False)
    pd.DataFrame(
        {"station_id": frame["station_id"].unique(), "latitude": 46.0, "longitude": 9.0}
    ).to_csv(tmp_path / "stations.csv", index=False)
    config = {
        "adapter": "csv",
        "dataset_id": "external-test",
        "observations": "observations.csv",
        "stations": "stations.csv",
        "timezone": timezone,
        "cadence_minutes": cadence,
        "variables": variables
        or {"temperature": {"column": "reading", "unit": "degC"}},
    }
    path = tmp_path / "import.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, tmp_path / "processed", config


def test_adapter_is_actually_abstract():
    with pytest.raises(TypeError):
        DatasetAdapter()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "identifier", ["001", "NA", "a_b", "a/b", "München", "../outside"]
)
def test_station_ids_are_reversible_safe_and_not_truncated(identifier):
    key = station_key(identifier)
    assert "_" not in key and "/" not in key
    assert bytes.fromhex(key[1:]).decode() == identifier


@pytest.mark.parametrize(
    "variable,unit,values,expected",
    [
        ("temperature", "K", [273.15, 283.15], [0, 10]),
        ("temperature", "degF", [32, 50], [0, 10]),
        ("humidity", "fraction", [0.5, 1], [50, 100]),
        ("pressure", "Pa", [100000, 90000], [1000, 900]),
        ("wind_speed", "km h-1", [3.6, 7.2], [1, 2]),
        ("wind_speed", "knots", [1, 2], [1852 / 3600, 3704 / 3600]),
        ("wind_direction", "radian", [np.pi, np.pi / 2], [180, 90]),
        ("precipitation", "m", [0.001, 0], [1, 0]),
    ],
)
def test_units_are_converted_before_bounds(tmp_path, variable, unit, values, expected):
    frame = pd.DataFrame(
        {
            "station_id": ["001", "001"],
            "timestamp": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "reading": values,
        }
    )
    spec = {"column": "reading", "unit": unit}
    if variable == "precipitation":
        spec.update(precipitation_kind="interval_total", timestamp_position="start")
    path, output, _ = _inputs(tmp_path, frame=frame, variables={variable: spec})
    manifest = import_dataset(path, output)
    with xr.open_dataset(output / manifest["outputs"][0]["path"]) as ds:
        np.testing.assert_allclose(ds["value"].values, expected, atol=1e-6)
        assert ds.attrs["original_station_id"] == "001"
        assert ds.attrs["coordinate_source"] == "user_station_catalogue_wgs84"
        assert variable not in ds  # no invented auxiliary product


def test_missing_flags_are_provider_specific_before_hourly_average(tmp_path):
    frame = pd.DataFrame(
        {
            "station_id": "NA",
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="15min").astype(
                str
            ),
            "reading": [10, 20, 0, 999],
            "quality": [151, 255, "bad", "good"],
        }
    )
    spec = {
        "column": "reading",
        "unit": "degC",
        "quality_column": "quality",
        "invalid_quality_values": ["bad"],
        "missing_values": [999],
    }
    path, output, _ = _inputs(
        tmp_path, frame=frame, cadence=15, variables={"temperature": spec}
    )
    result = import_dataset(path, output)
    with xr.open_dataset(output / result["outputs"][0]["path"]) as ds:
        assert ds["value"].item() == 15
        assert ds["sample_count"].item() == 4
        assert ds["valid_sample_count"].item() == 2
        assert ds.attrs["original_station_id"] == "NA"


def test_rain_requires_all_expected_samples_and_end_labels_shift(tmp_path):
    times = pd.date_range("2024-01-01T00:15", periods=16, freq="15min")
    frame = pd.DataFrame(
        {
            "station_id": "rain_A",
            "timestamp": times.astype(str),
            "reading": [1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 0, 0, 1, -999, 1, 1],
        }
    )
    frame = frame.drop(index=5)  # one interval not received, not a zero
    spec = {
        "column": "reading",
        "unit": "mm",
        "precipitation_kind": "interval_total",
        "timestamp_position": "end",
        "missing_values": [-999],
    }
    path, output, _ = _inputs(
        tmp_path, frame=frame, cadence=15, variables={"precipitation": spec}
    )
    result = import_dataset(path, output)
    with xr.open_dataset(output / result["outputs"][0]["path"]) as ds:
        np.testing.assert_allclose(
            ds["value"].values, [10, np.nan, 0, np.nan], equal_nan=True
        )
        assert ds["time"].values[0] == np.datetime64("2024-01-01T00", "ns")


def test_circular_average_and_hourly_reindexing(tmp_path):
    frame = pd.DataFrame(
        {
            "station_id": "wind",
            "timestamp": [
                "2024-01-01T00:00",
                "2024-01-01T00:30",
                "2024-01-01T02:00",
                "2024-01-01T02:30",
            ],
            "reading": [350, 10, 0, 180],
        }
    )
    path, output, _ = _inputs(
        tmp_path,
        frame=frame,
        cadence=30,
        variables={"wind_direction": {"column": "reading", "unit": "degree"}},
    )
    result = import_dataset(path, output)
    with xr.open_dataset(output / result["outputs"][0]["path"]) as ds:
        np.testing.assert_allclose(
            ds["value"].values, [0, np.nan, np.nan], atol=1e-6, equal_nan=True
        )


@pytest.mark.parametrize(
    "timestamp,timezone,expected",
    [
        ("2024-01-01T01:00", "Europe/Rome", "2024-01-01T00:00"),
        ("2024-01-01T01:00+01:00", "UTC", "2024-01-01T00:00"),
    ],
)
def test_explicit_timezone_normalization(tmp_path, timestamp, timezone, expected):
    frame = pd.DataFrame(
        {"station_id": ["a"], "timestamp": [timestamp], "reading": [1]}
    )
    path, output, _ = _inputs(tmp_path, frame=frame, timezone=timezone)
    result = import_dataset(path, output)
    with xr.open_dataset(output / result["outputs"][0]["path"]) as ds:
        assert ds["time"].values[0] == np.datetime64(expected, "ns")


@pytest.mark.parametrize(
    "change,match",
    [
        ({"cadence_minutes": 120}, "cadence"),
        ({"timezone": ""}, "ZoneInfo"),
        ({"unknown_option": 1}, "Unknown"),
        ({"variables": {"ozone": {"column": "reading", "unit": "ppb"}}}, "Unsupported"),
        (
            {"variables": {"temperature": {"column": "reading", "unit": "fahrenheit"}}},
            "Unsupported unit",
        ),
        (
            {"variables": {"precipitation": {"column": "reading", "unit": "mm"}}},
            "interval_total",
        ),
    ],
)
def test_invalid_config_does_not_publish(tmp_path, change, match):
    path, output, config = _inputs(tmp_path)
    config.update(change)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises((ValueError, KeyError), match=match):
        import_dataset(path, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".giano-import-*"))


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2024-01-01T00:00", "2024-01-01T00:00"],
        ["2024-01-01T00:15", "2024-01-01T01:15"],
        ["2024-10-27T02:00", "2024-10-27T03:00"],
        ["2024-03-31T02:00", "2024-03-31T03:00"],
    ],
)
def test_ambiguous_duplicate_or_off_grid_timestamps_are_rejected(tmp_path, timestamps):
    frame = pd.DataFrame({"station_id": "a", "timestamp": timestamps, "reading": 1})
    path, output, _ = _inputs(tmp_path, frame=frame, timezone="Europe/Rome")
    with pytest.raises(ValueError):
        import_dataset(path, output)
    assert not output.exists()


def test_bad_coordinates_and_unexplained_missing_codes_are_rejected(tmp_path):
    path, output, _ = _inputs(tmp_path)
    catalogue = pd.read_csv(tmp_path / "stations.csv")
    catalogue["latitude"] = 91
    catalogue.to_csv(tmp_path / "stations.csv", index=False)
    with pytest.raises(ValueError, match="latitude"):
        import_dataset(path, output)
    catalogue["latitude"] = 46
    catalogue.to_csv(tmp_path / "stations.csv", index=False)
    frame = pd.read_csv(tmp_path / "observations.csv", dtype=str)
    frame.loc[0, "reading"] = "broken_sensor"
    frame.to_csv(tmp_path / "observations.csv", index=False)
    with pytest.raises(ValueError, match="missing_values"):
        import_dataset(path, output)
    assert not output.exists()


def test_manifest_validation_cli_and_no_overwrite(tmp_path, capsys):
    path, output, _ = _inputs(tmp_path)
    assert main(["import", "--config", str(path), "--output-dir", str(output)]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert len(manifest["sources"]) == 3
    assert manifest["sources"][0]["sha256"] == file_sha256(path)
    before = file_sha256(output / "manifest.json")
    with pytest.raises(FileExistsError):
        import_dataset(path, output)
    assert before == file_sha256(output / "manifest.json")
    assert main(["validate", "--data-dir", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == manifest["validation"]


def test_foreign_dataset_loads_and_giano_preserves_observations(tmp_path):
    path, output, _ = _inputs(tmp_path)
    frame = pd.read_csv(tmp_path / "observations.csv")
    frame.loc[24:31, "reading"] = np.nan
    other = frame.copy()
    other["station_id"] = "different/station"
    pd.concat([frame, other]).to_csv(tmp_path / "observations.csv", index=False)
    pd.DataFrame(
        {
            "station_id": ["station_with_underscores", "different/station"],
            "latitude": [46, 46.1],
            "longitude": [9, 9.1],
        }
    ).to_csv(tmp_path / "stations.csv", index=False)
    manifest = import_dataset(path, output)
    panel = build_spatiotemporal_panel(output, "temperature")
    assert panel.values.shape == (120, 2)
    model_config = ImputeFormerConfig(
        input_embedding_dim=8,
        spatial_embedding_dim=8,
        num_layers=1,
        num_heads=2,
        projection_tokens=4,
        feed_forward_dim=16,
    )
    model = MeteorologicalImputeFormer(model_config)
    checkpoint = tmp_path / "tiny.pt"
    torch.save(
        {
            "model_class": "MeteorologicalImputeFormer",
            "variable": "temperature",
            "model_config": asdict(model_config),
            "state_dict": model.state_dict(),
            "training_config": asdict(TrainingConfig(seq_len=24, max_nodes=2)),
        },
        checkpoint,
    )
    reconstructed = impute_variable(
        output,
        checkpoint,
        tmp_path / "imputed",
        device=torch.device("cpu"),
        fallback="none",
    )
    assert len(reconstructed) == 2
    for source, destination in zip(
        sorted(manifest["outputs"], key=lambda item: item["path"]),
        reconstructed,
        strict=True,
    ):
        with (
            xr.open_dataset(output / source["path"]) as original,
            xr.open_dataset(destination) as filled,
        ):
            np.testing.assert_array_equal(
                original["value"].values, filled["value"].values
            )
            observed = np.isfinite(original["value"].values)
            np.testing.assert_array_equal(
                filled["imputed_value"].values[observed],
                original["value"].values[observed],
            )
            assert (filled["window_prediction_count"].values[~observed] > 0).all()
            assert np.isfinite(filled["imputed_value"].values).all()


def test_validator_rejects_duplicate_station_across_storage_groups(tmp_path):
    path, output, _ = _inputs(tmp_path)
    result = import_dataset(path, output)
    source = output / result["outputs"][0]["path"]
    (output / "second").mkdir()
    (output / "second" / source.name).write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        validate_processed_dataset(output)


def test_foreign_training_benchmark_and_completed_resume(tmp_path):
    frame = pd.DataFrame(
        {
            "station_id": "foreign_01",
            "timestamp": pd.date_range("2024-01-01", periods=720, freq="h").astype(str),
            "reading": 10 + np.sin(np.arange(720) / 8),
        }
    )
    path, output, _ = _inputs(tmp_path, frame=frame)
    import_dataset(path, output)
    checkpoint = tmp_path / "new-weights" / "temperature.pt"
    model = ImputeFormerConfig(
        input_embedding_dim=4,
        spatial_embedding_dim=4,
        num_heads=2,
        num_layers=1,
        projection_tokens=2,
        feed_forward_dim=8,
    )
    config = TrainingConfig(
        seq_len=24,
        max_nodes=2,
        batch_size=2,
        max_epochs=1,
        max_batches_per_epoch=1,
        max_validation_batches=1,
    )
    result = train_variable(
        output,
        "temperature",
        checkpoint,
        model,
        config,
        torch.device("cpu"),
        progress_every=0,
    )
    digest = file_sha256(checkpoint)
    resumed = train_variable(
        output,
        "temperature",
        checkpoint,
        model,
        config,
        torch.device("cpu"),
        resume=True,
        progress_every=0,
    )
    assert result == resumed
    assert digest == file_sha256(checkpoint)
    comparison = benchmark_checkpoint(
        checkpoint,
        output,
        "val",
        mode="block",
        block_length=6,
        max_batches=1,
        device=torch.device("cpu"),
    )
    assert comparison["count"] > 0
    assert np.isfinite(comparison["model_mae"])


def test_meteotrentino_adapter_delegates_to_corrected_preprocessing(tmp_path):
    csv_dir, nc_dir = tmp_path / "raw", tmp_path / "era5"
    csv_dir.mkdir()
    nc_dir.mkdir()
    times = pd.date_range("2024-01-01", periods=8, freq="15min")
    pd.DataFrame(
        {
            "time": times,
            "value": [10, 0, 20, 999, 11, 12, 13, 14],
            "quality": [1, 151, 1, 1, 1, 1, 1, 1],
        }
    ).to_csv(csv_dir / "T0001_temperature.csv", index=False)
    xr.Dataset(
        {"t2m": ("time", [283.15, 284.15], {"units": "K"})},
        coords={
            "time": pd.date_range("2024-01-01", periods=2, freq="h"),
            "latitude": 46.0,
            "longitude": 11.0,
        },
    ).to_netcdf(nc_dir / "T0001_temperature.nc", engine="h5netcdf")
    config = tmp_path / "meteo.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "adapter": "meteotrentino",
                "dataset_id": "meteo-test",
                "raw_csv": "raw",
                "raw_era5": "era5",
            }
        )
    )
    output = tmp_path / "processed"
    manifest = import_dataset(config, output)
    with xr.open_dataset(output / manifest["outputs"][0]["path"]) as ds:
        np.testing.assert_allclose(ds["value"].values, [15, 12.5])
        np.testing.assert_allclose(ds["temperature"].values, [10, 11])
        assert ds.attrs["preprocessing_version"] == "raw-filter-before-hourly-v2"
    assert (
        "timezone_not_verified_by_importer"
        in manifest["validation"]["files"][0]["warnings"]
    )


def test_noaa_preparer_keeps_missing_and_calm_semantics(tmp_path):
    import gzip
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "noaa_example", Path(__file__).parents[2] / "tools/prepare_noaa_isd_lite.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "isd-history.csv").write_text(
        "USAF,WBAN,STATION NAME,LAT,LON\n067700,99999,LUGANO,46.004,8.911\n"
    )
    with gzip.open(raw / "067700-99999-2024.gz", "wt") as handle:
        handle.write(
            "2024 1 1 0 -9999 0 10130 0 0 0 -1 -9999\n2024 1 1 1 100 0 10130 360 10 0 0 0\n"
        )
    destination = tmp_path / "csv"
    module.prepare(raw, destination, 2024, ["067700-99999"])
    result = pd.read_csv(destination / "observations.csv")
    assert pd.isna(result.loc[0, "temperature"])
    assert pd.isna(result.loc[0, "wind_direction"])
    assert result.loc[0, "wind_speed"] == 0
    assert result.loc[1, "temperature"] == 10
    assert result.loc[1, "wind_speed"] == 1
    assert result.loc[1, "wind_direction"] == 360
    assert "sea_level_pressure" not in result
    with pytest.raises(FileExistsError):
        module.prepare(raw, destination, 2024, ["067700-99999"])
