"""Import foreign station datasets without changing existing data or weights."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from giano.data_processing.adapters.base import DatasetAdapter
from giano.data_processing.adapters.csv import CsvDatasetAdapter
from giano.data_processing.adapters.meteotrentino import MeteoTrentinoAdapter
from giano.meteorology import PHYSICAL_BOUNDS_BY_VARIABLE, UNITS_BY_VARIABLE
from giano.netcdf import open_dataset_robust
from giano.provenance import file_sha256
from giano.variables import VARIABLE_TYPE_NAMES

CSV_KEYS = {
    "adapter",
    "dataset_id",
    "observations",
    "stations",
    "timezone",
    "cadence_minutes",
    "variables",
    "delimiter",
    "timestamp_column",
    "station_column",
}
METEO_KEYS = {"adapter", "dataset_id", "raw_csv", "raw_era5"}


def load_adapter(path: Path) -> tuple[DatasetAdapter, dict[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("dataset_id"), str)
        or not config["dataset_id"].strip()
    ):
        raise ValueError("Import config requires a nonempty dataset_id")
    kind = config.get("adapter")
    allowed = (
        CSV_KEYS if kind == "csv" else METEO_KEYS if kind == "meteotrentino" else set()
    )
    if not allowed or set(config) - allowed:
        raise ValueError(
            "Unknown adapter or config keys. Supported adapters: csv, meteotrentino"
        )
    required = (
        {"observations", "stations", "timezone", "cadence_minutes", "variables"}
        if kind == "csv"
        else {"raw_csv", "raw_era5"}
    )
    if not required.issubset(config):
        raise ValueError(f"Adapter {kind} requires {sorted(required)}")
    if kind == "csv":
        return CsvDatasetAdapter(config, path.resolve().parent), config
    return MeteoTrentinoAdapter(
        path.resolve().parent / config["raw_csv"],
        path.resolve().parent / config["raw_era5"],
    ), config


def validate_processed_dataset(root: Path) -> dict[str, Any]:
    """Validate on disk, one series at a time."""
    paths = sorted(root.glob("*/*_merged.nc"))
    if not paths:
        raise ValueError(
            f"No files at {root}/<storage-group>/<station>_<variable>_merged.nc"
        )
    reports = []
    seen = set()
    spans: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for path in paths:
        variable = next(
            (
                name
                for name in VARIABLE_TYPE_NAMES
                if path.name.endswith(f"_{name}_merged.nc")
            ),
            None,
        )
        if variable is None:
            raise ValueError(f"Unsupported variable in {path}")
        station = path.name.removesuffix(f"_{variable}_merged.nc")
        if not re.fullmatch(r"[A-Za-z0-9-]+", station):
            raise ValueError(
                f"Unsafe/ambiguous station filename: {path.name}; use the CSV importer for arbitrary IDs"
            )
        if (station, variable) in seen:
            raise ValueError(f"Duplicate station/variable: {station}/{variable}")
        seen.add((station, variable))
        with open_dataset_robust(path) as ds:
            if (
                "time" not in ds.coords
                or "value" not in ds
                or ds["value"].dims != ("time",)
            ):
                raise ValueError(f"{path}: requires time and value(time)")
            if ds["time"].dtype.kind != "M":
                raise ValueError(f"{path}: time must be decoded datetimes")
            times = pd.DatetimeIndex(ds["time"].values).as_unit("ns")
            time_ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            if (
                not len(times)
                or times.hasnans
                or times.has_duplicates
                or not times.is_monotonic_increasing
                or np.any(time_ns % (3600 * 10**9))
                or np.any(np.diff(time_ns) != 3600 * 10**9)
            ):
                raise ValueError(
                    f"{path}: timestamps must form a complete unique hourly grid. Missing observations are NaN rows"
                )
            for coordinate, limit in (
                ("station_latitude", 90),
                ("station_longitude", 180),
            ):
                raw = ds.attrs.get(coordinate, ds.coords.get(coordinate, np.nan))
                coordinates = np.asarray(raw, dtype=float)
                if (
                    coordinates.size != 1
                    or not np.isfinite(coordinates).all()
                    or np.any(np.abs(coordinates) > limit)
                ):
                    raise ValueError(f"{path}: invalid {coordinate}")
            values = np.asarray(ds["value"].values, dtype=float)
            low, high = PHYSICAL_BOUNDS_BY_VARIABLE[variable]
            if np.isinf(values).any() or np.any(
                np.isfinite(values) & ((values < low) | (values > high))
            ):
                raise ValueError(f"{path}: unfiltered physical-range violations")
            if ds["value"].attrs.get("units") != UNITS_BY_VARIABLE[variable]:
                raise ValueError(
                    f"{path}: value units must be {UNITS_BY_VARIABLE[variable]}"
                )
            for auxiliary in (variable, "wind_u", "wind_v"):
                if auxiliary in ds and ds[auxiliary].dims != ("time",):
                    raise ValueError(
                        f"{path}: auxiliary {auxiliary} must align to time"
                    )
            warnings = []
            if len(times) < 72:
                warnings.append("shorter_than_release_window_72h")
            if not np.isfinite(values).any():
                warnings.append("no_valid_observations")
            if not str(ds.attrs.get("timestamp_convention", "")).startswith("UTC;"):
                warnings.append("timezone_not_verified_by_importer")
            reports.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    "station_id": station,
                    "original_station_id": str(
                        ds.attrs.get("original_station_id", station)
                    ),
                    "variable": variable,
                    "hours": len(times),
                    "observed": int(np.isfinite(values).sum()),
                    "missing": int(np.isnan(values).sum()),
                    "warnings": warnings,
                }
            )
            spans.setdefault(variable, []).append((times[0], times[-1]))
    panels = {}
    for variable, bounds in spans.items():
        hours = (
            int(
                (max(end for _, end in bounds) - min(start for start, _ in bounds))
                / pd.Timedelta(hours=1)
            )
            + 1
        )
        panels[variable] = {
            "stations": len(bounds),
            "hours": hours,
            "cells": hours * len(bounds),
        }
    return {
        "schema_version": 1,
        "files": reports,
        "panels": panels,
        "note": "Contract validity is not model accuracy or sufficient training coverage. Inspect per-series warnings and panel memory before running.",
    }


def import_dataset(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Publish all-or-nothing into a new dataset root."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"Output already exists: {output_dir}; choose a new directory"
        )
    adapter, config = load_adapter(config_path)
    sources = [config_path.resolve(), *adapter.source_paths()]
    identities = [
        {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in sources
    ]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".giano-import-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / "processed"
        staging.mkdir()
        adapter.write(staging)
        report = validate_processed_dataset(staging)
        if identities != [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in sources
        ]:
            raise RuntimeError("Inputs changed during import.")
        manifest = {
            "schema_version": 1,
            "dataset_id": config["dataset_id"],
            "adapter": config["adapter"],
            "created_at": datetime.now(UTC).isoformat(),
            "config": config,
            "sources": identities,
            "validation": report,
            "outputs": [
                {"path": entry["file"], "sha256": file_sha256(staging / entry["file"])}
                for entry in report["files"]
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Output appeared during import: {output_dir}")
        staging.rename(output_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import a dataset into a new processed root, or validate an existing one."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    importer = subparsers.add_parser(
        "import", help="convert configured CSV/Meteotrentino input"
    )
    importer.add_argument("--config", type=Path, required=True)
    importer.add_argument("--output-dir", type=Path, required=True)
    validator = subparsers.add_parser(
        "validate", help="read-only processed-contract checks"
    )
    validator.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (
            import_dataset(args.config, args.output_dir)
            if args.command == "import"
            else validate_processed_dataset(args.data_dir)
        )
    except (ValueError, OSError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
