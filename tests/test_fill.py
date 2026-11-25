import numpy as np
import pytest
import torch
import xarray as xr

from giano.data import load_input
from giano.fill import fill, main


@pytest.fixture
def network(tmp_path):
    torch.set_num_threads(2)
    times = np.arange(
        np.datetime64("2023-12-25T00"),
        np.datetime64("2024-01-10T00"),
        np.timedelta64(1, "h"),
    ).astype("datetime64[ns]")
    data = np.full((len(times), 1), 10.1234567890123)
    hole = (times >= np.datetime64("2023-12-31T22")) & (
        times < np.datetime64("2024-01-01T03")
    )
    directions = np.full_like(data, 90.0)
    speeds = np.ones_like(data)
    data[hole] = np.nan
    directions[hole] = np.nan
    speeds[hole] = 0
    ds = xr.Dataset(
        coords={
            "time": times,
            "station": ["new-id"],
            "latitude": ("station", [46.0]),
            "longitude": ("station", [9.0]),
        }
    )
    for var, values, unit in (
        ("temperature", data, "degC"),
        ("wind_speed", speeds, "m s-1"),
        ("wind_direction", directions, "degree"),
    ):
        ds[var] = (("time", "station"), values, {"units": unit})
    return ds


def test_portable_file_roundtrip_over_new_year(tmp_path, network):
    raw = tmp_path / "source.nc"
    network.to_netcdf(raw, engine="h5netcdf")
    target = tmp_path / "filled.nc"
    args = [
        "--input",
        str(raw),
        "--start",
        "2023-12-31",
        "--end-exclusive",
        "2024-01-02",
        "--output",
        str(target),
    ]
    assert main(args) == 0
    with xr.open_dataset(target) as out:
        assert out.attrs["model"] == "Giano"
        expected = network.temperature.sel(time=out.time).values
        np.testing.assert_equal(out.temperature.values, expected)
        observed = np.isfinite(expected)
        np.testing.assert_equal(
            out.temperature_filled.values[observed], expected[observed]
        )
        assert np.isfinite(out.temperature_filled.values[~observed]).all()
        assert (out.temperature_method.values[~observed] == "point-pair").all()
        assert (out.wind_direction_method.values[~observed] == "calm-wind").all()
        assert np.isnan(out.wind_direction_filled.values[~observed]).all()
    with pytest.raises(SystemExit):
        main(args)


def test_contemporary_only_leaves_historical_fills_empty(network):
    result = fill(
        network, "2023-12-31", "2024-01-02", ["temperature"], contemporary_only=True
    )
    missing = ~np.isfinite(result.temperature.values)
    assert missing.sum() == 5
    assert (result.temperature_method.values[missing] == "historical-reference").all()
    assert np.isnan(result.temperature_filled.values[missing]).all()


def test_unknown_station_calibration_and_incompatible_units(tmp_path, network):
    with pytest.raises(ValueError):
        fill(network, "2023-12-31", "2024-01-02", ["temperature"], trentino=True)
    network.temperature.attrs["units"] = "K"
    path = tmp_path / "bad.nc"
    network.to_netcdf(path, engine="h5netcdf")
    with pytest.raises(ValueError, match="degC"):
        load_input(path, None)


def test_csv_hourly_grid_preserves_station_id_and_missing_row(tmp_path):
    source = tmp_path / "data.csv"
    catalogue = tmp_path / "stations.csv"
    catalogue.write_text("station_id,latitude,longitude\n001,45,9\n")
    source.write_text(
        "station_id,timestamp,temperature\n001,2024-01-01T00:00:00Z,2\n001,2024-01-01T02:00:00Z,3\n"
    )
    result = load_input(source, catalogue)
    assert result.station.values.tolist() == ["001"]
    np.testing.assert_equal(result.temperature.values[:, 0], [2, np.nan, 3])
