"""Regressions for invalid raw samples contaminating hourly ground truth."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from giano.data_processing.unify_dataset import (
    _resample_station_frame,
    _station_dataset,
    process_matched_pair,
)
from giano.meteorology import apply_physical_bounds


def _frame(values, quality=None):
    frame = pd.DataFrame(
        {"value": values},
        index=pd.date_range("2021-07-07 14:00", periods=len(values), freq="15min"),
    )
    if quality is not None:
        frame["quality"] = quality
    return frame


def test_pressure_invalid_sample_is_removed_before_mean():
    hourly = _resample_station_frame(
        _frame([842.9, 842.9, 0, 842.9], [1, 1, 145, 1]), "pressure"
    )
    row = hourly.iloc[0]
    assert row["value"] == pytest.approx(842.9)
    assert row["sample_count"] == 4
    assert row["valid_sample_count"] == 3
    assert row["valid_sample_fraction"] == 0.75
    assert row["quality"] == 1
    assert row["raw_quality_max"] == 145
    dataset = _station_dataset(hourly, "pressure")
    assert dataset.value.attrs["units"] == "hPa"
    assert dataset.attrs["preprocessing_version"] == "raw-filter-before-hourly-v2"
    assert "not temporal coverage" in dataset.valid_sample_fraction.attrs["description"]


@pytest.mark.parametrize("code", [151, 255])
def test_missing_quality_code_excludes_even_plausible_raw_value(code):
    hourly = _resample_station_frame(
        _frame([20, 0, 20, 20], [1, code, 1, 1]), "temperature"
    )
    assert hourly.value.iloc[0] == 20
    assert hourly.quality.iloc[0] == 1


@pytest.mark.parametrize(
    "values, expected",
    [([350, 999, 10, np.nan], 0), ([0, 180], np.nan), ([999, -1], np.nan)],
)
def test_direction_filters_before_circular_mean_and_rejects_undefined(values, expected):
    actual = _resample_station_frame(_frame(values), "wind_direction").value.iloc[0]
    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == pytest.approx(expected)


def test_rain_requires_all_received_samples_valid_but_does_not_invent_cadence():
    frame = _frame([0.2, 0.3, np.nan, 0.1, 0, 0, 0, 0, 0.2, 0.3, 0.4, 0.1])
    hourly = _resample_station_frame(frame, "precipitation")
    assert np.isnan(hourly.value.iloc[0])
    assert hourly.value.iloc[1] == 0
    assert hourly.value.iloc[2] == pytest.approx(1)
    sparse = frame.iloc[[0, 8]]
    hourly = _resample_station_frame(sparse, "precipitation")
    assert np.isnan(hourly.value.iloc[1])
    assert hourly.sample_count.iloc[1] == 0
    # One received valid sample is not evidence of full temporal coverage.
    assert hourly.value.iloc[0] == 0.2
    assert hourly.valid_sample_fraction.iloc[0] == 1


@pytest.mark.parametrize("kind", ["duplicate", "unsorted", "missing"])
def test_invalid_timestamps_rejected(kind):
    frame = _frame([10, 20])
    if kind == "duplicate":
        frame.index = pd.DatetimeIndex([frame.index[0], frame.index[0]])
    elif kind == "unsorted":
        frame = frame.iloc[::-1]
    else:
        frame.index = pd.DatetimeIndex([frame.index[0], pd.NaT])
    with pytest.raises(ValueError, match="timestamps"):
        _resample_station_frame(frame, "temperature")


def test_matched_era5_path_persists_filtered_hour_and_provenance(tmp_path):
    frame = _frame(
        [842.9, 842.9, 0, 842.9, 844, 844, 844, 844], [1, 1, 145, 1, 1, 1, 1, 1]
    )
    frame.index.name = "time"
    csv = tmp_path / "T0366_pressure.csv"
    frame.to_csv(csv)
    nc = tmp_path / "T0366_pressure.nc"
    xr.Dataset(
        {"sp": ("time", [84300.0, 84400.0], {"units": "Pa"})},
        coords={
            "time": pd.date_range("2021-07-07 14:00", periods=2, freq="h"),
            "latitude": 46.0,
            "longitude": 11.0,
        },
    ).to_netcdf(nc, engine="h5netcdf")
    output = process_matched_pair("T0366", "pressure", str(csv), str(nc), tmp_path)
    with xr.open_dataset(output, engine="h5netcdf") as result:
        assert result.value.values[0] == pytest.approx(842.9)
        assert result.value.attrs["units"] == "hPa"
        assert result.pressure.attrs["units"] == "hPa"
        assert result.pressure.values[0] == 843
        assert result.valid_sample_count.values[0] == 3
        assert result.attrs["coordinate_source"] == "auxiliary_grid_proxy"
        assert result.attrs["preprocessing_version"] == "raw-filter-before-hourly-v2"


def test_nonmissing_quality_codes_do_not_discard_physical_values() -> None:
    values = np.array([15.0, 20.0, 500.0])
    quality = np.array([1.0, 145.0, 145.0])

    bounded = apply_physical_bounds(values, "temperature", quality)

    assert bounded[0] == 15.0
    assert bounded[1] == 20.0
    assert np.isnan(bounded[2])


def test_explicit_missing_quality_codes_are_masked() -> None:
    temperature = apply_physical_bounds(
        np.array([0.0, 12.5, 9.0]),
        "temperature",
        np.array([151.0, 1.0, 140.0]),
    )
    pressure = apply_physical_bounds(
        np.array([800.0]),
        "pressure",
        np.array([255.0]),
    )

    assert np.isnan(temperature[0])
    assert temperature[1:].tolist() == [12.5, 9.0]
    assert np.isnan(pressure[0])
