"""Split merged data into train, validation, and test sets.

This script takes transient merged NetCDF files and splits them into train,
validation, and test sets based on station IDs,
ensuring data integrity and balanced distribution across sets.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default split configuration
DEFAULT_SPLIT_CONFIG = {
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42,
}


def load_split_config(config_path: Path) -> dict[str, Any]:  # noqa: PLR0911
    """Load and validate split configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with config_path.open() as f:
            raw_cfg = yaml.safe_load(f) or {}
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Failed to read config: {config_path}") from exc

    if not isinstance(raw_cfg, dict):
        raise ValueError(
            f"Config {config_path} top-level must be a mapping, got "
            f"{type(raw_cfg).__name__}",
        )

    split_cfg = raw_cfg.get("split", {})
    if not isinstance(split_cfg, dict):
        raise ValueError(f"Config {config_path}: 'split' must be a mapping")

    defaults = DEFAULT_SPLIT_CONFIG

    try:
        cfg = {
            "train_ratio": float(split_cfg.get("train_ratio", defaults["train_ratio"])),
            "val_ratio": float(split_cfg.get("val_ratio", defaults["val_ratio"])),
            "test_ratio": float(split_cfg.get("test_ratio", defaults["test_ratio"])),
            "random_seed": int(split_cfg.get("random_seed", defaults["random_seed"])),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid split values in {config_path}") from exc

    ratios = (cfg["train_ratio"], cfg["val_ratio"], cfg["test_ratio"])
    if any(not (0.0 <= r <= 1.0) for r in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Invalid split ratios in {config_path}: {ratios}")

    return cfg


def extract_station_id(filename: str) -> str:
    """Extract station ID from filename.

    Parameters
    ----------
    filename : str
        Filename in format "TXXXX_variable_merged.nc"

    Returns
    -------
    str
        Station ID (e.g., "T0008")

    """
    return filename.split("_", maxsplit=1)[0] if "_" in filename else filename


def extract_variable_name(filename: str) -> str | None:
    """Extract variable name from filename.

    Example: T0008_temperature_merged.nc -> temperature
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 2:  # noqa: PLR2004
        return None

    # Expected layout: TXXXX_<variable>_merged
    # Keep support for variable names with underscores (e.g. wind_speed).
    variable_parts = parts[1:-1] if parts[-1] == "merged" else parts[1:]

    if not variable_parts:
        return None
    return "_".join(variable_parts)


def get_unique_stations(data_folder: Path) -> list[str]:
    """Get list of unique station IDs from data folder.

    Parameters
    ----------
    data_folder : Path
        Path to folder containing merged NetCDF files

    Returns
    -------
    list[str]
        Sorted list of unique station IDs

    """
    station_ids = set()

    for file_path in data_folder.glob("*_merged.nc"):
        station_id = extract_station_id(file_path.name)
        station_ids.add(station_id)

    return sorted(station_ids)


def split_stations(  # noqa: PLR0913
    station_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    data_folder: Path | None = None,
) -> dict[str, list[str]]:
    """Split station IDs into train/validation/test sets.

    When *data_folder* is provided the split is stratified by variable
    coverage so that rare variables (e.g. wind) appear in all splits.

    Parameters
    ----------
    station_ids : list[str]
        List of station IDs to split
    train_ratio : float
        Fraction of stations for training
    val_ratio : float
        Fraction of stations for validation
    test_ratio : float
        Fraction of stations for testing
    random_seed : int
        Random seed for reproducible splits
    data_folder : Path | None
        Source folder with *_merged.nc files (enables stratification).

    Returns
    -------
    dict[str, list[str]]
        Dictionary with keys 'train', 'val', 'test' and lists of station IDs

    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        error_msg = f"Ratios sum to {total_ratio}, not 1.0"
        raise ValueError(error_msg)
    if not station_ids:
        msg = "station_ids must not be empty"
        raise ValueError(msg)

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_seed)

    # Build station → variable-set mapping for stratification
    station_vars: dict[str, frozenset[str]] = {}
    if data_folder is not None and data_folder.exists():
        for f in data_folder.glob("*_merged.nc"):
            sid = extract_station_id(f.name)
            vname = extract_variable_name(f.name)
            if vname:
                existing = station_vars.get(sid, frozenset())
                station_vars[sid] = existing | {vname}

    def add_station(
        station_id: str,
        target_set: set[str],
        coverage: set[str],
    ) -> None:
        target_set.add(station_id)
        coverage.update(station_vars.get(station_id, frozenset()))

    def pick_unassigned_station(
        candidates: list[str],
        assigned: set[str],
    ) -> str | None:
        for sid in candidates:
            if sid not in assigned:
                return sid
        return None

    if station_vars:
        n_stations = len(station_ids)
        target_train = int(n_stations * train_ratio)
        target_val = int(n_stations * val_ratio)
        target_test = n_stations - target_train - target_val

        if n_stations >= 3:
            if target_val < 1:
                target_val = 1
            if target_test < 1:
                target_test = 1
            while target_train + target_val + target_test > n_stations:
                if target_train >= max(target_val, target_test) and target_train > 1:
                    target_train -= 1
                elif target_val > 1:
                    target_val -= 1
                else:
                    target_test -= 1

        target_sizes = {
            "train": target_train,
            "val": target_val,
            "test": target_test,
        }

        split_sets: dict[str, set[str]] = {name: set() for name in target_sizes}
        split_coverage: dict[str, set[str]] = {name: set() for name in target_sizes}
        assigned: set[str] = set()

        variable_to_stations: dict[str, list[str]] = defaultdict(list)
        for sid in station_ids:
            for variable_name in station_vars.get(sid, frozenset()):
                variable_to_stations[variable_name].append(sid)

        for variable_name, stations_for_var in sorted(
            variable_to_stations.items(),
            key=lambda item: (len(item[1]), item[0]),
        ):
            shuffled_stations = stations_for_var.copy()
            rng.shuffle(shuffled_stations)
            required_splits = ("train", "val", "test")[: len(shuffled_stations)]
            for split_name in required_splits:
                coverage = split_coverage[split_name]
                if variable_name in coverage:
                    continue
                chosen = pick_unassigned_station(shuffled_stations, assigned)
                if chosen is not None:
                    add_station(chosen, split_sets[split_name], coverage)
                    assigned.add(chosen)

        shuffled_ids = station_ids.copy()
        rng.shuffle(shuffled_ids)
        for sid in shuffled_ids:
            if sid in assigned:
                continue

            deficits = {
                name: target_size - len(split_sets[name])
                for name, target_size in target_sizes.items()
            }
            positive_deficits = {
                split_name: deficit
                for split_name, deficit in deficits.items()
                if deficit > 0
            }

            if positive_deficits:
                target_split = max(
                    positive_deficits,
                    key=lambda split_name: positive_deficits[split_name],
                )
            else:
                target_split = min(
                    ("train", "val", "test"),
                    key=lambda split_name: (
                        len(split_sets[split_name]),
                        split_name != "train",
                    ),
                )

            add_station(sid, split_sets[target_split], split_coverage[target_split])
            assigned.add(sid)

        for variable_name, stations_for_var in variable_to_stations.items():
            required = ("train", "val", "test")[: len(stations_for_var)]
            missing = [
                split_name
                for split_name in required
                if not (split_sets[split_name] & set(stations_for_var))
            ]
            if missing:
                raise ValueError(
                    f"Cannot preserve {variable_name} coverage in splits: {missing}",
                )

        splits = {name: sorted(stations) for name, stations in split_sets.items()}
    else:
        # Fallback: simple shuffle split
        shuffled_ids = station_ids.copy()
        rng.shuffle(shuffled_ids)

        n_stations = len(shuffled_ids)
        n_train = int(n_stations * train_ratio)
        n_val = int(n_stations * val_ratio)

        splits = {
            "train": sorted(shuffled_ids[:n_train]),
            "val": sorted(shuffled_ids[n_train : n_train + n_val]),
            "test": sorted(shuffled_ids[n_train + n_val :]),
        }

    n_stations = len(station_ids)
    logger.info("Dataset split:")
    for name, label in (("train", "Train"), ("val", "Validation"), ("test", "Test")):
        count = len(splits[name])
        logger.info(
            "  %s: %d stations (%.1f%%)", label, count, count / n_stations * 100
        )

    return splits


def copy_files_by_stations(
    source_folder: Path,
    target_folder: Path,
    station_ids: list[str],
    set_name: str,
) -> int:
    """Copy files for given station IDs to target folder.

    Parameters
    ----------
    source_folder : Path
        Source folder containing merged NetCDF files
    target_folder : Path
        Target folder for copied files
    station_ids : list[str]
        List of station IDs to copy
    set_name : str
        Name of the dataset split (train/val/test)

    Returns
    -------
    int
        Number of files copied

    """
    target_folder.mkdir(parents=True, exist_ok=True)

    files_copied = 0

    for station_id in station_ids:
        # Find all files for this station
        station_files = list(source_folder.glob(f"{station_id}_*_merged.nc"))

        for source_file in station_files:
            target_file = target_folder / source_file.name
            shutil.copy2(source_file, target_file)
            files_copied += 1

    logger.info("Copied %d files to %s set", files_copied, set_name)
    return files_copied


def run_split(
    source_folder: Path,
    output_folder: Path,
    split_config: dict[str, Any],
) -> int:
    """Build and atomically publish train/validation/test directories."""
    if not source_folder.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")
    station_ids = get_unique_stations(source_folder)
    if not station_ids:
        raise ValueError("No stations found in source folder")
    split_info = split_stations(
        station_ids,
        train_ratio=split_config["train_ratio"],
        val_ratio=split_config["val_ratio"],
        test_ratio=split_config["test_ratio"],
        random_seed=split_config["random_seed"],
        data_folder=source_folder,
    )
    output_folder.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".processed-staging-", dir=output_folder.parent)
    )
    try:
        total_files_copied = 0
        for set_name, stations in split_info.items():
            total_files_copied += copy_files_by_stations(
                source_folder,
                staging / set_name,
                stations,
                set_name,
            )
        source_count = len(list(source_folder.glob("*_merged.nc")))
        if total_files_copied != source_count:
            raise RuntimeError(
                f"Split copied {total_files_copied}/{source_count} source files",
            )
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "split": split_info,
            "file_count": total_files_copied,
            "config": split_config,
        }
        source_manifest = source_folder / "manifest.json"
        if source_manifest.is_file():
            preprocessing = json.loads(source_manifest.read_text(encoding="utf-8"))
            manifest["preprocessing_version"] = preprocessing.get(
                "preprocessing_version"
            )
            shutil.copy2(source_manifest, staging / "preprocessing_manifest.json")
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        backup = output_folder.with_name(f".{output_folder.name}.previous")
        if backup.exists():
            shutil.rmtree(backup)
        if output_folder.exists():
            output_folder.rename(backup)
        try:
            staging.rename(output_folder)
        except OSError:
            if backup.exists() and not output_folder.exists():
                backup.rename(output_folder)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return total_files_copied
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: list[str] | None = None) -> int:
    """Split the dataset into train/validation/test."""
    parser = argparse.ArgumentParser(description="Split merged NetCDFs by station")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file with split settings (default: config.yaml)",
    )
    parser.add_argument("--source-folder", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, default=None)
    args = parser.parse_args(argv)

    split_config = load_split_config(args.config)
    with args.config.open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    paths = raw_config.get("paths", {}) if isinstance(raw_config, dict) else {}
    if not isinstance(paths, dict):
        raise ValueError("Configuration 'paths' section must be a mapping")
    source_folder = args.source_folder
    output_folder = args.output_folder or Path(
        paths.get("processed", "data/2-processed-v2"),
    )

    logger.info("Starting dataset split process...")

    total_files_copied = run_split(source_folder, output_folder, split_config)

    logger.info("Dataset split completed successfully!")
    logger.info("Total files copied: %d", total_files_copied)
    logger.info("Output directory: %s", output_folder.absolute())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
