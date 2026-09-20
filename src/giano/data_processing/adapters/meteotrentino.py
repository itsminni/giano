"""Reuse the corrected Meteotrentino implementation without duplicating it."""

from pathlib import Path

from giano.data_processing.adapters.base import DatasetAdapter
from giano.data_processing.unify_dataset import run_unification


class MeteoTrentinoAdapter(DatasetAdapter):
    """Preserve raw-filter-before-hourly-v2 and its explicit metadata limits."""

    def __init__(self, csv_dir: Path, nc_dir: Path) -> None:
        self.csv_dir = csv_dir
        self.nc_dir = nc_dir

    def source_paths(self) -> list[Path]:
        return sorted([*self.csv_dir.glob("*.csv"), *self.nc_dir.glob("*.nc")])

    def write(self, destination: Path) -> list[Path]:
        return run_unification(destination / "all", self.csv_dir, self.nc_dir)
