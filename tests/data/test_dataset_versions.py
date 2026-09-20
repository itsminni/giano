"""Dataset version comparison uses temporal splits and never mutates data."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from giano.data_processing.compare_versions import compare_versions, file_sha256
from giano.data_processing.split_dataset import run_split


def _write(root, values, *, time_shift=0):
    directory = root / "test"
    directory.mkdir(parents=True)
    path = directory / "T0001_temperature_merged.nc"
    xr.Dataset(
        {"value": ("time", values, {"units": "degC"})},
        coords={
            "time": pd.date_range("2020-01-01", periods=20, freq="h")
            + pd.Timedelta(hours=time_shift)
        },
        attrs={
            "preprocessing_version": "raw-filter-before-hourly-v2",
            "station_latitude": 46.0,
            "station_longitude": 11.0,
        },
    ).to_netcdf(path, engine="h5netcdf")
    return path


def test_version_audit_counts_global_temporal_splits_and_preserves_files(tmp_path):
    old, new = np.arange(20, dtype=float), np.arange(20, dtype=float)
    old[15] = np.nan
    new[2] += 2
    new[18] = np.nan
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    old_path = _write(old_root, old)
    new_path = _write(new_root, new)
    before = [file_sha256(old_path), file_sha256(new_path)]
    report = compare_versions(old_root, new_root)
    scores = report["variables"]["temperature"]
    assert scores["train"]["changed_finite"] == 1
    assert scores["val"]["recovered"] == 1
    assert scores["test"]["newly_missing"] == 1
    assert scores["all"]["max_abs_change"] == 2
    assert report["storage_groups_unchanged"]
    assert report["files"][0]["old_sha256"] == before[0]
    assert [file_sha256(old_path), file_sha256(new_path)] == before
    json.dumps(report, allow_nan=False)


def test_version_audit_rejects_time_shift(tmp_path):
    _write(tmp_path / "old", np.arange(20, dtype=float))
    _write(tmp_path / "new", np.arange(20, dtype=float), time_shift=1)
    with pytest.raises(ValueError, match="time axes"):
        compare_versions(tmp_path / "old", tmp_path / "new")


def test_split_preserves_preprocessing_manifest(tmp_path):
    path = _write(tmp_path / "source", np.arange(20, dtype=float))
    source = path.parent
    preprocessing = {
        "preprocessing_version": "raw-filter-before-hourly-v2",
        "sources": [],
    }
    (source / "manifest.json").write_text(json.dumps(preprocessing))
    output = tmp_path / "published"
    run_split(
        source,
        output,
        {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15, "random_seed": 42},
    )
    assert (
        json.loads((output / "preprocessing_manifest.json").read_text())
        == preprocessing
    )
    assert (
        json.loads((output / "manifest.json").read_text())["preprocessing_version"]
        == preprocessing["preprocessing_version"]
    )
