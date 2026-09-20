from __future__ import annotations

from pathlib import Path

import pytest

from giano.data_processing.split_dataset import split_stations


def test_rare_variable_is_present_in_every_split(tmp_path: Path) -> None:
    stations = [f"T{i:04d}" for i in range(9)]
    for station in stations:
        (tmp_path / f"{station}_temperature_merged.nc").touch()
    for station in stations[:3]:
        (tmp_path / f"{station}_wind_speed_merged.nc").touch()

    split = split_stations(stations, random_seed=12, data_folder=tmp_path)

    wind_stations = set(stations[:3])
    assert all(set(split[name]) & wind_stations for name in ("train", "val", "test"))


@pytest.mark.parametrize(
    ("coverage", "expected"),
    [
        (
            {},
            {"train": [1, 2, 3, 5, 6, 7], "val": [0], "test": [4, 8]},
        ),
        (
            {"temperature": list(range(9)), "wind_speed": [0, 1, 2]},
            {"train": [1, 4, 5, 6, 7, 8], "val": [2], "test": [0, 3]},
        ),
        (
            {
                "temperature": list(range(9)),
                "wind_speed": [0, 1],
                "humidity": [3, 4, 5, 6, 7],
                "pressure": [7],
            },
            {"train": [1, 2, 3, 5, 7, 8], "val": [0, 4], "test": [6]},
        ),
    ],
    ids=["unstratified", "rare-wind", "overlapping-rare-variables"],
)
def test_storage_split_preserves_recorded_assignments(
    tmp_path: Path,
    coverage: dict[str, list[int]],
    expected: dict[str, list[int]],
) -> None:
    """Keep the pre-refactor station identities, not just the split sizes."""
    stations = [f"T{i:04d}" for i in range(9)]
    for variable, indices in coverage.items():
        for index in indices:
            (tmp_path / f"{stations[index]}_{variable}_merged.nc").touch()
    result = split_stations(stations, random_seed=12, data_folder=tmp_path)
    assert result == {
        name: [stations[index] for index in indices]
        for name, indices in expected.items()
    }


@pytest.mark.parametrize("station_count", [1, 2, 3])
def test_small_network_keeps_coverage_without_duplicate_stations(
    tmp_path: Path, station_count: int
) -> None:
    stations = [f"T{i:04d}" for i in range(station_count)]
    for station in stations:
        (tmp_path / f"{station}_temperature_merged.nc").touch()
    result = split_stations(stations, data_folder=tmp_path)
    assigned = [station for group in result.values() for station in group]
    assert sorted(assigned) == stations
    assert all(result[name] for name in ("train", "val", "test")[:station_count])
