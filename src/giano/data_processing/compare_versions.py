"""Audit a separately regenerated dataset without changing either version."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from giano.netcdf import open_dataset_robust
from giano.provenance import file_sha256, git_provenance

NS_PER_HOUR = 3_600_000_000_000


def _index(root: Path) -> dict[str, Path]:
    paths = sorted(root.glob("*/*_merged.nc"))
    indexed = {path.name: path for path in paths}
    if not paths or len(indexed) != len(paths):
        raise ValueError(f"Empty dataset or duplicate station-variable files: {root}")
    return indexed


def _counts(
    old: np.ndarray, new: np.ndarray, tolerance: float
) -> dict[str, int | float]:
    old_finite, new_finite = np.isfinite(old), np.isfinite(new)
    common = old_finite & new_finite
    delta = np.abs(new[common] - old[common])
    return {
        "old_observed": int(old_finite.sum()),
        "new_observed": int(new_finite.sum()),
        "changed_finite": int((delta > tolerance).sum()),
        "recovered": int((~old_finite & new_finite).sum()),
        "newly_missing": int((old_finite & ~new_finite).sum()),
        "max_abs_change": float(delta.max()) if delta.size else 0.0,
    }


def compare_versions(
    old_root: Path, new_root: Path, *, tolerance: float = 1e-5
) -> dict[str, Any]:
    """Require matching time axes and report deltas on global temporal splits."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if old_root.resolve() == new_root.resolve():
        raise ValueError("Dataset versions must have distinct roots")
    old_index, new_index = _index(old_root), _index(new_root)
    if old_index.keys() != new_index.keys():
        raise ValueError("Dataset versions have different station-variable file sets")
    bounds: dict[str, tuple[int, int]] = {}
    for name, path in old_index.items():
        variable = name.split("_", 1)[1].removesuffix("_merged.nc")
        with open_dataset_robust(path) as ds:
            times = np.asarray(ds.time.values).astype("datetime64[ns]").astype(np.int64)
        first, last = bounds.get(variable, (int(times[0]), int(times[-1])))
        bounds[variable] = min(first, int(times[0])), max(last, int(times[-1]))

    aggregates: dict[str, dict[str, dict[str, int | float]]] = defaultdict(dict)
    records: list[dict[str, Any]] = []
    for name, old_path in old_index.items():
        new_path = new_index[name]
        variable = name.split("_", 1)[1].removesuffix("_merged.nc")
        with open_dataset_robust(old_path) as old, open_dataset_robust(new_path) as new:
            times = (
                np.asarray(old.time.values).astype("datetime64[ns]").astype(np.int64)
            )
            new_times = (
                np.asarray(new.time.values).astype("datetime64[ns]").astype(np.int64)
            )
            if not np.array_equal(times, new_times) or np.any(
                np.diff(times) != NS_PER_HOUR
            ):
                raise ValueError(f"Different or non-hourly time axes: {name}")
            old_values, new_values = (
                np.asarray(old.value.values),
                np.asarray(new.value.values),
            )
            if old_values.shape != times.shape or new_values.shape != times.shape:
                raise ValueError(f"Non-scalar station values: {name}")
            if np.isinf(new_values).any():
                raise ValueError(f"Infinite observations: {name}")
            coordinates_unchanged = all(
                old.attrs.get(key) == new.attrs.get(key)
                for key in ("station_latitude", "station_longitude")
            )
            auxiliary_unchanged = all(
                key in new
                and np.array_equal(old[key].values, new[key].values, equal_nan=True)
                for key in (variable, "wind_u", "wind_v")
                if key in old
            )
            first, last = bounds[variable]
            length = (last - first) // NS_PER_HOUR + 1
            positions = (times - first) // NS_PER_HOUR
            train_end, val_end = int(length * 0.7), int(length * (0.7 + 0.15))
            partitions = {
                "all": np.ones(len(times), dtype=bool),
                "train": positions < train_end,
                "val": (positions >= train_end) & (positions < val_end),
                "test": positions >= val_end,
            }
            per_split = {}
            for split, mask in partitions.items():
                counts = _counts(old_values[mask], new_values[mask], tolerance)
                per_split[split] = counts
                total = aggregates[variable].setdefault(
                    split, {key: 0 for key in counts}
                )
                for key, value in counts.items():
                    total[key] = (
                        max(total[key], value)
                        if key == "max_abs_change"
                        else total[key] + value
                    )
            records.append(
                {
                    "file": name,
                    "variable": variable,
                    "old_path": str(old_path),
                    "new_path": str(new_path),
                    "old_sha256": file_sha256(old_path),
                    "new_sha256": file_sha256(new_path),
                    "storage_group_unchanged": old_path.parent.name
                    == new_path.parent.name,
                    "coordinates_unchanged": coordinates_unchanged,
                    "auxiliary_unchanged": auxiliary_unchanged,
                    "preprocessing_version": new.attrs.get("preprocessing_version"),
                    "units": new.value.attrs.get("units"),
                    "splits": per_split,
                }
            )
    return {
        "schema_version": 1,
        "old_root": str(old_root.resolve()),
        "new_root": str(new_root.resolve()),
        "file_count": len(records),
        "absolute_tolerance_physical_units": tolerance,
        "temporal_split_ratios": [0.7, 0.15, 0.15],
        "time_axes_unchanged": True,
        "storage_groups_unchanged": all(
            row["storage_group_unchanged"] for row in records
        ),
        "coordinates_unchanged": all(row["coordinates_unchanged"] for row in records),
        "auxiliary_unchanged": all(row["auxiliary_unchanged"] for row in records),
        "variables": dict(aggregates),
        "files": records,
        "code": git_provenance(Path.cwd().resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    """Write a strict JSON audit to a new file only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists, choose a new audit path")
    report = compare_versions(args.old_root, args.new_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "files"}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
