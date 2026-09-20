"""Full-history exports preserve every hour and distinguish unsupported gaps."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from giano.provenance import file_sha256


@pytest.fixture(scope="module")
def history():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
        yield importlib.import_module("prepare_giano_history")


def _records(tmp_path, auxiliary=False):
    times = np.arange(
        np.datetime64("2019-12-31T00", "ns"),
        np.datetime64("2020-01-02T00", "ns"),
        np.timedelta64(1, "h"),
    )
    values = np.arange(len(times), dtype=float) / 7
    values[23:26] = np.nan
    filled = values.copy()
    filled[23:25] = 3.25
    support = np.zeros(len(times), dtype=np.uint16)
    support[23:25] = 2
    source = tmp_path / "data/train/T0001_temperature_merged.nc"
    source.parent.mkdir(parents=True)
    original = xr.Dataset({"value": ("time", values)}, coords={"time": times})
    original.value.attrs["units"] = "degC"
    if auxiliary:
        signal = np.arange(len(times), dtype=np.float32) / 2
        signal[24] = np.nan
        original["temperature"] = ("time", signal)
    original.to_netcdf(source, engine="h5netcdf")
    directory = tmp_path / "inference/temperature"
    directory.mkdir(parents=True)
    result = original.copy(deep=True)
    result["imputed_value"] = ("time", filled)
    result["window_prediction_count"] = ("time", support)
    reconstructed = directory / f"{source.stem}_imputed.nc"
    result.to_netcdf(reconstructed, engine="h5netcdf")
    (directory / "temperature_manifest.json").write_text(
        json.dumps(
            {
                "model_family": "imputeformer",
                "fallback": "none",
                "checkpoint_sha256": "test",
            }
        )
    )
    record = {
        "status": "complete",
        "request": {
            "fallback": "none",
            "seed": 42,
            "checkpoint_sha256": "test",
            "dataset": {
                "files": [
                    {"path": "train/" + source.name, "sha256": file_sha256(source)}
                ]
            },
        },
        "files": {p.name: file_sha256(p) for p in directory.iterdir()},
    }
    (directory / "complete.json").write_text(json.dumps(record))
    return source, reconstructed, values, filled


def _presentation(root, history):
    features = []
    for station in ("T0001", "T0002"):
        card = {
            "id": station,
            "geometry": {"type": "Point", "coordinates": [11, 46]},
            "variables": {"temperature": {"unit": "degC", "example": None}}
            if station == "T0001"
            else {},
        }
        relative = f"stations/{station}.json"
        history.write_json(root / relative, card)
        features.append(
            {
                "id": station,
                "geometry": card["geometry"],
                "properties": {"card": relative},
            }
        )
    history.write_json(root / "stations.geojson", {"features": features})
    history.write_json(root / "content.json", {"en": {}, "it": {}})
    (root / "README.md").write_text("# Giano\n")
    history.write_json(
        root / "manifest.json",
        {
            "kind": "giano_station_presentation",
            "counts": {"stations": 2, "examples": 0},
            "files": {
                p.relative_to(root).as_posix(): file_sha256(p)
                for p in root.rglob("*")
                if p.is_file()
            },
        },
    )


@pytest.mark.parametrize("auxiliary", [False, True])
def test_complete_history_roundtrip_year_boundary_nulls_and_station_links(
    tmp_path, history, monkeypatch, auxiliary
):
    monkeypatch.setattr(history, "VARIABLE_TYPE_NAMES", ("temperature",))
    _, _, values, filled = _records(tmp_path, auxiliary=auxiliary)
    _presentation(tmp_path / "presentation", history)
    bundle = tmp_path / "bundle"
    history.build(
        tmp_path / "presentation", bundle, tmp_path / "data", tmp_path / "inference"
    )
    checks = history.verify_histories(bundle, tmp_path / "data", tmp_path / "inference")
    assert not (bundle / "README.md").exists()
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert "README.md" not in manifest["files"]
    assert "stations/T0001.json" in manifest["files"]
    assert "histories/T0001/temperature/2020.json" in manifest["files"]
    assert all("\\" not in name for name in manifest["files"])
    assert (tmp_path / "presentation/README.md").read_text() == "# Giano\n"
    assert checks["hours"] == 48 and checks["annual_chunks"] == 2
    assert checks["observed_hours"] == 45
    assert checks["reconstructed_hours"] == 2 and checks["unfilled_hours"] == 1
    assert checks["stations_with_history"] == 1
    chunks = [
        json.loads((bundle / f"histories/T0001/temperature/{year}.json").read_text())
        for year in (2019, 2020)
    ]
    assert chunks[0]["observed"][0] == 0
    assert chunks[0]["giano_estimates"][0] is None
    if auxiliary:
        assert chunks[0]["era5_land"][0] == 0
        assert chunks[0]["era5_land"][-1] == 11.5
        assert chunks[1]["era5_land"][:2] == [None, 12.5]
    else:
        assert all("era5_land" not in chunk for chunk in chunks)
    actual = [x for chunk in chunks for x in chunk["observed"]]
    np.testing.assert_array_equal(np.asarray(actual, dtype=float), values)
    merged = [
        o if o is not None else g
        for chunk in chunks
        for o, g in zip(chunk["observed"], chunk["giano_estimates"], strict=True)
    ]
    np.testing.assert_array_equal(np.asarray(merged, dtype=float), filled)
    index = json.loads((bundle / "histories/T0001/temperature/index.json").read_text())
    assert index["ground_truth_in_natural_gaps"] is None
    assert [row["month"] for row in index["monthly_overview"]] == ["2019-12", "2020-01"]


def test_example_era5_uses_exact_source_hours_and_preserves_missing_values(history):
    times = np.array(
        ["2020-01-01T00", "2020-01-01T01", "2020-01-01T02"], dtype="datetime64[s]"
    )
    values = np.array([0.0, np.nan, 3.0])
    assert history.example_era5(times, values, [str(t) for t in times]) == [0, None, 3]
    assert history.example_era5(times, values, [str(times[-1])]) == [3]
    for timestamp in ("2019-12-31T23", "2020-01-01T00:30", "2020-01-01T03"):
        with pytest.raises(ValueError, match="timestamps"):
            history.example_era5(times, values, [timestamp])


def test_export_adds_era5_without_changing_example_predictions(tmp_path, history):
    source, reconstructed, _, _ = _records(tmp_path, auxiliary=True)
    example_path = "examples/T0001/temperature.json"
    original = {
        "series": {
            "timestamps": [
                "2019-12-31T23:00:00",
                "2020-01-01T00:00:00",
                "2020-01-01T01:00:00",
            ],
            "reconstruction": [3.25, 3.25, None],
        },
        "metrics": {"mae": 0.5},
    }
    bundle = tmp_path / "bundle"
    history.write_json(bundle / example_path, original)
    history.export_history(
        {
            "station": "T0001",
            "variable": "temperature",
            "source": source,
            "reconstructed": reconstructed,
            "source_sha256": file_sha256(source),
            "checkpoint_sha256": "test",
        },
        bundle,
        example_path,
    )
    result = json.loads((bundle / example_path).read_text())
    assert result["series"].pop("era5_land") == [11.5, None, 12.5]
    assert result == original


@pytest.mark.parametrize("mutation", ["observed", "unsupported_estimate", "timestamp"])
def test_history_rejects_changed_observations_or_unsupported_estimates(
    tmp_path, history, mutation
):
    source, output, _, _ = _records(tmp_path)
    with xr.open_dataset(output, engine="h5netcdf") as opened:
        changed = opened.load()
    if mutation == "observed":
        changed["imputed_value"].values[1] += 1
    elif mutation == "unsupported_estimate":
        changed["imputed_value"].values[25] = 9
    else:
        changed = changed.assign_coords(time=changed.time + np.timedelta64(1, "h"))
    changed.to_netcdf(output, engine="h5netcdf")
    with pytest.raises(ValueError):
        history.load_history(source, output)


def test_monthly_wind_direction_is_circular_and_undefined_mean_is_null(history):
    circular = history.stats(np.array([359.0, 1.0, np.nan]), "wind_direction")
    assert abs((circular["mean"] + 180) % 360 - 180) < 1e-10
    assert circular["mean_kind"] == "circular" and circular["count"] == 2
    assert history.stats(np.array([0.0, 180.0]), "wind_direction")["mean"] is None
    assert history.stats(np.array([np.nan]), "temperature")["mean"] is None


def test_full_leap_year_is_not_truncated(history):
    times = np.arange(
        np.datetime64("2020-01-01T00"),
        np.datetime64("2021-01-01T00"),
        np.timedelta64(1, "h"),
    )
    periods = list(history.period_slices(times, "Y"))
    assert periods == [("2020", slice(0, 8784))]
    assert len(list(history.period_slices(times, "M"))) == 12


@pytest.mark.parametrize("mutation", ["output", "extra_file", "request"])
def test_completed_inference_refuses_changed_cache(tmp_path, history, mutation):
    _, output, _, _ = _records(tmp_path)
    record = history.verify_completed(output.parent)
    expected = record["request"].copy()
    if mutation == "output":
        output.write_bytes(b"changed")
    elif mutation == "extra_file":
        (output.parent / "unexpected.txt").write_text("extra")
    else:
        expected["checkpoint_sha256"] = "different"
    with pytest.raises(ValueError):
        history.verify_completed(output.parent, expected)


def test_resume_skips_verified_variable_without_running_model(
    tmp_path, history, monkeypatch
):
    _, output, _, _ = _records(tmp_path)
    runner = importlib.import_module("generate_giano_history")
    record = history.verify_completed(output.parent)
    monkeypatch.setattr(runner, "request_identity", lambda *args: record["request"])

    def unexpected(*args, **kwargs):
        pytest.fail("Completed inference should not be rerun")

    monkeypatch.setattr(runner, "impute_variable", unexpected)
    assert (
        runner.generate(
            tmp_path / "data",
            tmp_path / "unused.pt",
            tmp_path / "inference",
            "temperature",
        )
        == record
    )


@pytest.mark.parametrize("field", ["observed", "era5_land"])
def test_deep_verification_detects_changed_values_even_if_hashes_are_updated(
    tmp_path, history, monkeypatch, field
):
    monkeypatch.setattr(history, "VARIABLE_TYPE_NAMES", ("temperature",))
    _records(tmp_path, auxiliary=True)
    _presentation(tmp_path / "presentation", history)
    bundle = tmp_path / "bundle"
    history.build(
        tmp_path / "presentation", bundle, tmp_path / "data", tmp_path / "inference"
    )
    relative = "histories/T0001/temperature/2020.json"
    path = bundle / relative
    chunk = json.loads(path.read_text())
    chunk[field][4] += 0.125
    history.write_json(path, chunk)
    index_path = bundle / "histories/T0001/temperature/index.json"
    index = json.loads(index_path.read_text())
    index["chunks"][1].update(
        {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    )
    history.write_json(index_path, index)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][relative] = file_sha256(path)
    manifest["files"][index_path.relative_to(bundle).as_posix()] = file_sha256(
        index_path
    )
    history.write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Changed exported"):
        history.verify_histories(bundle, tmp_path / "data", tmp_path / "inference")
