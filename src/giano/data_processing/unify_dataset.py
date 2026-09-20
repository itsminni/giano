"""Helpers to unify CSV station dumps with ERA5 NetCDF files into merged NetCDFs.

This module indexes CSV and NetCDF data, pairs matching station/variable files,
aligns their time ranges, merges them with xarray, and writes merged NetCDFs to
a caller-provided temporary directory.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from giano.meteorology import (
    UNITS_BY_VARIABLE,
    apply_physical_bounds,
    derive_wind_direction_from_uv,
    derive_wind_speed_from_uv,
    normalize_era5_units,
)
from giano.netcdf import find_time_coord, open_dataset_robust

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATION_RE = re.compile(r"T\d{4}")  # es. T0008, T1234

# Directories
CSV_DIR = Path("data/1-raw/dump")
NC_DIR = Path("data/1-raw/era5")


# Map ERA5 variable names to CSV-friendly names
VARIABLE_MAP = {
    "t2m": "temperature",
    "d2m": "dewpoint",
    "tp": "precipitation",
    "total_precipitation": "precipitation",
    "msl": "pressure",
    "sp": "pressure",
    "q": "specific_humidity",
    "rh": "humidity",
    "u10": "wind_u",
    "v10": "wind_v",
    "ws10": "wind_speed",
    "wd10": "wind_direction",
}


def parse_filename(path: str) -> tuple[str, str]:
    """Return the station_id and variable regardless of order.

    Args:
        path: file name or path containing a station code like 'T0008'.

    Returns:
        A (station_id, variable) pair.

    """
    name = Path(path).stem
    m = STATION_RE.search(name)
    if not m:
        msg = f"ID station not found in {path}"
        raise ValueError(msg)
    station = m.group(0)
    variable = name.replace(station, "").strip("_-").lower()
    return station, variable


# Map CSV variable names to NC file group names when they differ
CSV_TO_NC_GROUP = {
    "wind_speed": "wind",
    "wind_direction": "wind",
    "wind": "wind",
    "temperature": "temperature",
    "precipitation": "precipitation",
}


def _first_finite_scalar(raw_value: object) -> float | None:
    """Extract the first finite scalar from xarray-like coordinate values."""
    try:
        arr = np.asarray(raw_value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    finite = arr.reshape(-1)[np.isfinite(arr.reshape(-1))]
    if finite.size == 0:
        return None
    return float(finite[0])


def _extract_station_coordinates(ds: xr.Dataset) -> tuple[float, float] | None:
    """Return station coordinates from dataset coords / attrs when available."""
    latitude_candidates = (
        "latitude",
        "lat",
        "station_latitude",
        "station_lat",
        "y",
    )
    longitude_candidates = (
        "longitude",
        "lon",
        "station_longitude",
        "station_lon",
        "x",
    )

    latitude = None
    longitude = None
    for name in latitude_candidates:
        if name in ds.coords:
            latitude = _first_finite_scalar(ds.coords[name].values)
        elif name in ds.data_vars:
            latitude = _first_finite_scalar(ds[name].values)
        elif name in ds.attrs:
            latitude = _first_finite_scalar(ds.attrs[name])
        if latitude is not None:
            break

    for name in longitude_candidates:
        if name in ds.coords:
            longitude = _first_finite_scalar(ds.coords[name].values)
        elif name in ds.data_vars:
            longitude = _first_finite_scalar(ds[name].values)
        elif name in ds.attrs:
            longitude = _first_finite_scalar(ds.attrs[name])
        if longitude is not None:
            break

    if latitude is None or longitude is None:
        return None
    return float(latitude), float(longitude)


def _build_station_coordinate_index(
    nc_index: dict[tuple[str, str], str],
) -> dict[str, tuple[float, float]]:
    """Map station ids to coordinates using any available ERA5 file."""
    coordinate_index: dict[str, tuple[float, float]] = {}
    for (station, _var), nc_path in sorted(nc_index.items()):
        if station in coordinate_index:
            continue
        try:
            with open_dataset_robust(nc_path) as ds:
                coords = _extract_station_coordinates(ds)
        except (OSError, RuntimeError, ValueError):
            logger.debug("Failed to scan coordinates from %s", nc_path, exc_info=True)
            continue
        if coords is not None:
            coordinate_index[station] = coords
    return coordinate_index


def _attach_station_coordinates(
    ds: xr.Dataset,
    station_coordinates: tuple[float, float] | None,
) -> xr.Dataset:
    """Persist station coordinates as dataset attrs for downstream loaders."""
    if station_coordinates is None:
        return ds
    ds_out = ds.copy()
    latitude, longitude = station_coordinates
    ds_out.attrs["station_latitude"] = float(latitude)
    ds_out.attrs["station_longitude"] = float(longitude)
    return ds_out


def _resample_station_frame(df: pd.DataFrame, variable_name: str) -> pd.DataFrame:
    """Filter raw samples before aggregating.

    Counts measure received samples, not temporal coverage.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Station data requires a DatetimeIndex")
    if (
        df.index.hasnans
        or df.index.has_duplicates
        or not df.index.is_monotonic_increasing
    ):
        raise ValueError("Station timestamps must be unique, valid and sorted")
    if "value" not in df:
        raise ValueError("Station data requires a value column")
    resampled_columns: dict[str, pd.Series] = {}
    quality = pd.to_numeric(df["quality"], errors="coerce") if "quality" in df else None
    value_series = pd.Series(
        apply_physical_bounds(
            pd.to_numeric(df["value"], errors="coerce").to_numpy(),
            variable_name,
            None if quality is None else quality.to_numpy(),
        ),
        index=df.index,
    )
    samples = value_series.resample("1h").size()
    valid_samples = value_series.resample("1h").count()
    resampled_columns["sample_count"] = samples
    resampled_columns["valid_sample_count"] = valid_samples
    resampled_columns["valid_sample_fraction"] = valid_samples / samples.replace(
        0, np.nan
    )

    if "value" in df.columns:
        if variable_name == "precipitation":
            hourly = value_series.resample("1h").sum(min_count=1)
            resampled_columns["value"] = hourly.where(valid_samples == samples)
        elif variable_name == "wind_direction":
            radians = np.deg2rad(value_series.to_numpy())
            sin_series = pd.Series(np.sin(radians), index=value_series.index)
            cos_series = pd.Series(np.cos(radians), index=value_series.index)
            sin_component = sin_series.resample("1h").mean()
            cos_component = cos_series.resample("1h").mean()
            direction_values = (
                np.rad2deg(
                    np.arctan2(sin_component.to_numpy(), cos_component.to_numpy())
                )
                + 360.0
            ) % 360.0
            direction = pd.Series(direction_values, index=sin_component.index)
            undefined = np.hypot(sin_component, cos_component) < 1e-8
            direction[(sin_component.isna()) | (cos_component.isna()) | undefined] = (
                np.nan
            )
            resampled_columns["value"] = direction
        else:
            resampled_columns["value"] = value_series.resample("1h").mean()

    if quality is not None:
        resampled_columns["raw_quality_max"] = quality.resample("1h").max()
        resampled_columns["quality"] = (
            quality.where(value_series.notna()).resample("1h").max()
        )

    for column_name in df.columns:
        if column_name in {"value", "quality", *resampled_columns}:
            continue
        series = pd.to_numeric(df[column_name], errors="coerce")
        resampled_columns[column_name] = series.resample("1h").mean()

    output = pd.DataFrame(resampled_columns)
    output["value"] = apply_physical_bounds(output["value"].to_numpy(), variable_name)
    return output


def _station_dataset(frame: pd.DataFrame, variable: str) -> xr.Dataset:
    """Attach the hourly validity contract."""
    ds = frame.to_xarray()
    ds["value"].attrs["units"] = UNITS_BY_VARIABLE[variable]
    ds["value"].attrs["aggregation"] = (
        "sum_all_received_samples_valid"
        if variable == "precipitation"
        else "circular_mean_valid_samples"
        if variable == "wind_direction"
        else "mean_valid_samples"
    )
    ds["sample_count"].attrs["description"] = (
        "Number of received raw samples in the hour"
    )
    ds["valid_sample_count"].attrs["description"] = (
        "Raw samples passing bounds and missing-code checks"
    )
    ds["valid_sample_fraction"].attrs.update(
        {
            "description": "Valid / received samples; not temporal coverage",
            "units": "1",
        }
    )
    if "quality" in ds:
        ds["quality"].attrs["description"] = (
            "Maximum quality code among valid raw samples; not a quality ranking"
        )
        ds["raw_quality_max"].attrs["description"] = (
            "Maximum raw quality code, including excluded samples"
        )
    ds.attrs.update(
        {
            "preprocessing_version": "raw-filter-before-hourly-v2",
            "hourly_coverage_basis": "received_samples_only",
            "timestamp_convention": "source timestamps preserved",
            "coordinate_source": "auxiliary_grid_proxy",
        }
    )
    return ds


def _normalize_era5_dataarray(
    data_var: xr.DataArray,
    variable_name: str,
) -> xr.DataArray:
    """Convert known ERA5 units and update attrs for persisted merged files."""
    attrs = dict(getattr(data_var, "attrs", {}))
    raw_units = str(attrs.get("units") or attrs.get("GRIB_units") or "").strip().lower()
    normalized_values = normalize_era5_units(
        data_var.to_numpy(),
        variable_name,
        attrs,
    )

    updated_units = None
    if variable_name in {"temperature", "dewpoint"} and raw_units in {"k", "kelvin"}:
        updated_units = "C"
    elif variable_name == "pressure" and raw_units in {"pa", "pascal", "pascals"}:
        updated_units = "hPa"
    elif variable_name == "precipitation" and raw_units in {"m", "meter", "metre"}:
        updated_units = "mm"

    if updated_units is not None:
        attrs["units"] = updated_units

    normalized = xr.DataArray(
        normalized_values,
        coords=data_var.coords,
        dims=data_var.dims,
        attrs=attrs,
        name=variable_name,
    )
    return normalized


def _prepare_era5_dataset(ds_nc: xr.Dataset, variable_name: str) -> xr.Dataset:
    """Select and normalize the ERA5 signal relevant to the target variable."""
    rename_map = {}
    for nc_var in ds_nc.data_vars:
        lowered = str(nc_var).lower()
        if lowered in VARIABLE_MAP:
            rename_map[nc_var] = VARIABLE_MAP[lowered]
    if rename_map:
        ds_nc = ds_nc.rename(rename_map)

    if variable_name in ds_nc.data_vars:
        selected = ds_nc[[variable_name]].copy()
        selected[variable_name] = _normalize_era5_dataarray(
            selected[variable_name],
            variable_name,
        )
        return selected

    if variable_name in {"wind_speed", "wind_direction"} and {
        "wind_u",
        "wind_v",
    }.issubset(ds_nc.data_vars):
        wind_u = ds_nc["wind_u"]
        wind_v = ds_nc["wind_v"]
        derived_values = (
            derive_wind_speed_from_uv(wind_u.to_numpy(), wind_v.to_numpy())
            if variable_name == "wind_speed"
            else derive_wind_direction_from_uv(wind_u.to_numpy(), wind_v.to_numpy())
        )
        derived_attrs = {
            "units": "m s**-1" if variable_name == "wind_speed" else "degree",
        }
        derived = xr.DataArray(
            derived_values,
            coords=wind_u.coords,
            dims=wind_u.dims,
            attrs=derived_attrs,
            name=variable_name,
        )
        return xr.Dataset({variable_name: derived}, coords=ds_nc.coords)

    if len(ds_nc.data_vars) == 1:
        original_name = str(next(iter(ds_nc.data_vars)))
        selected = ds_nc.rename({original_name: variable_name})
        selected = selected[[variable_name]].copy()
        selected[variable_name] = _normalize_era5_dataarray(
            selected[variable_name],
            variable_name,
        )
        return selected

    logger.warning(
        '  Could not isolate ERA5 variable "%s" from vars=%s',
        variable_name,
        list(ds_nc.data_vars),
    )
    return xr.Dataset(coords=ds_nc.coords)


def build_file_index(data_dir: Path, pattern: str) -> dict[tuple[str, str], str]:
    """Build an index {(station_id, variable): path} for matching files."""
    file_index: dict[tuple[str, str], str] = {}
    if data_dir.exists():
        for p in sorted(data_dir.glob(pattern)):
            try:
                station, var = parse_filename(p.name)
                file_index[(station, var)] = str(p)
            except ValueError as e:  # noqa: PERF203
                logger.warning("Skipping %s (parse error): %s -> %s", p.name, p, e)
    else:
        logger.warning("Directory not found: %s", data_dir)
    return file_index


def build_matched_pairs(
    csv_index: dict[tuple[str, str], str],
    nc_index: dict[tuple[str, str], str],
) -> list[tuple[str, str, str, str]]:
    """Build matched (station, variable, csv_path, nc_path) tuples."""
    matched_pairs: list[tuple[str, str, str, str]] = []
    for (station, csv_var), csv_path in csv_index.items():
        # Candidate NC variable names to check
        candidates = []
        if csv_var in CSV_TO_NC_GROUP:
            candidates.append(CSV_TO_NC_GROUP[csv_var])
        candidates.append(csv_var)
        if "_" in csv_var:
            candidates.append(csv_var.split("_")[0])

        found = False
        for cand in candidates:
            key = (station, cand)
            if key in nc_index:
                matched_pairs.append((station, csv_var, csv_path, nc_index[key]))
                found = True
                break
        if not found:
            logger.info(
                "No matching NC file for CSV %s/%s "
                "— will be processed as CSV-only if applicable",
                station,
                csv_var,
            )
    return matched_pairs


def process_matched_pair(  # noqa: C901, PLR0912, PLR0915
    station: str,
    var: str,
    csv_path: str,
    nc_path: str,
    out_dir: Path,
) -> Path:
    """Merge a CSV station file with matching NC data and write merged output."""
    logger.info(
        "Processing %s / %s -> CSV: %s  NC: %s",
        station,
        var,
        csv_path,
        nc_path,
    )

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV missing: {csv_path}")

    try:
        df = pd.read_csv(csv_path, parse_dates=["time"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"CSV missing 'time' or cannot be parsed: {csv_path}") from exc

    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    df = df.set_index("time")
    df = _resample_station_frame(df, var)

    try:
        with open_dataset_robust(nc_path) as ds_nc_in:
            ds_nc = _prepare_era5_dataset(ds_nc_in, var)
            station_coordinates = _extract_station_coordinates(ds_nc_in)

            # Detect time coordinate name
            tname = find_time_coord(ds_nc)
            if tname is None:
                raise ValueError(f"NetCDF has no time-like coordinate: {nc_path}")
            if tname != "time":
                ds_nc = ds_nc.rename({tname: "time"})
                tname = "time"

            # Station-specific ERA5 files may retain singleton grid dimensions.
            # Drop only singleton axes; non-singleton grids are ambiguous and
            # must be reduced upstream to the station location.
            extra_dims = [dim for dim in ds_nc.dims if dim != tname]
            ambiguous = [dim for dim in extra_dims if ds_nc.sizes[dim] != 1]
            if ambiguous:
                raise ValueError(
                    f"ERA5 file has unresolved spatial dimensions {ambiguous}: {nc_path}",
                )
            if extra_dims:
                ds_nc = ds_nc.squeeze(extra_dims, drop=True)

            # Convert to pandas timestamps for reliable comparison
            ds_times = pd.to_datetime(ds_nc[tname].values)
            if ds_times.size == 0:
                raise ValueError(f"NetCDF time coordinate is empty: {nc_path}")
            if ds_times.has_duplicates or not ds_times.is_monotonic_increasing:
                raise ValueError(
                    f"ERA5 timestamps are duplicate or unsorted: {nc_path}"
                )
            if df.index.has_duplicates or not df.index.is_monotonic_increasing:
                raise ValueError(
                    f"Station timestamps are duplicate or unsorted: {csv_path}"
                )
            ds_time_min = ds_times.min()
            ds_time_max = ds_times.max()

            start = max(df.index.min(), ds_time_min)
            end = min(df.index.max(), ds_time_max)
            if start >= end:
                raise ValueError(
                    f"No overlapping period for {(station, var)}: {start}..{end}",
                )

            df_aligned = df.loc[start:end]
            ds_nc = ds_nc.sel({tname: slice(start, end)})

            # CSV -> xarray and merge
            ds_csv = _station_dataset(df_aligned, var)
            ds_csv = _attach_station_coordinates(ds_csv, station_coordinates)
            ds_csv, ds_nc = xr.align(ds_csv, ds_nc, join="inner", copy=False)
            csv_times = np.asarray(ds_csv["time"].values).astype("datetime64[ns]")
            era5_times = np.asarray(ds_nc["time"].values).astype("datetime64[ns]")
            if csv_times.size == 0 or not np.array_equal(csv_times, era5_times):
                raise ValueError(f"Failed exact time alignment for {station}/{var}")
            merged = xr.merge([ds_csv, ds_nc], join="exact", compat="no_conflicts")
            merged = _attach_station_coordinates(merged, station_coordinates)

            out = out_dir / f"{station}_{var}_merged.nc"
            merged.to_netcdf(
                out,
                encoding={v: {"zlib": True, "complevel": 4} for v in merged.data_vars},
            )
            logger.info("  %s created", out)
            return out
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Failed to merge {station}/{var}: {exc}") from exc


def process_csv_only_pairs(
    csv_index: dict[tuple[str, str], str],
    nc_index: dict[tuple[str, str], str],
    matched_pairs: list[tuple[str, str, str, str]],
    out_dir: Path,
    station_coordinate_index: dict[str, tuple[float, float]] | None = None,
) -> list[Path]:
    """Create merged files from CSV-only variables when NC is missing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_set = {(station, var) for station, var, _, _ in matched_pairs}
    outputs: list[Path] = []
    for (station, var), csv_path in csv_index.items():
        if var not in ("pressure", "humidity"):
            continue
        # Skip if this pair was already processed with an NC
        if (station, var) in matched_set or (station, var) in nc_index:
            continue

        logger.info(
            "CSV-only process %s / %s -> CSV: %s (no NC)",
            station,
            var,
            csv_path,
        )
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV missing: {csv_path}")
        try:
            df = pd.read_csv(csv_path, parse_dates=["time"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Cannot parse CSV time column: {csv_path}") from exc
        if df.empty:
            raise ValueError(f"CSV is empty: {csv_path}")
        df = df.set_index("time")
        df = _resample_station_frame(df, var)

        # Convert CSV -> xarray and save as merged NetCDF (CSV-only)
        ds_csv = _station_dataset(df, var)
        station_coordinates = (
            None
            if station_coordinate_index is None
            else station_coordinate_index.get(station)
        )
        ds_csv = _attach_station_coordinates(
            ds_csv,
            station_coordinates,
        )
        out = out_dir / f"{station}_{var}_merged.nc"
        ds_csv.to_netcdf(
            out,
            encoding={v: {"zlib": True, "complevel": 4} for v in ds_csv.data_vars},
        )
        logger.info("  %s created (CSV-only)", out)
        outputs.append(out)
    return outputs


def run_unification(
    out_dir: Path,
    csv_dir: Path = CSV_DIR,
    nc_dir: Path = NC_DIR,
) -> list[Path]:
    """Run full CSV+NC unification pipeline and save merged NetCDF files."""
    if not csv_dir.is_dir() or not nc_dir.is_dir():
        raise FileNotFoundError(
            f"Raw input directories are required: csv={csv_dir}, nc={nc_dir}",
        )
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    csv_index = build_file_index(csv_dir, "*.csv")
    nc_index = build_file_index(nc_dir, "*.nc")
    station_coordinate_index = _build_station_coordinate_index(nc_index)

    matched_pairs = build_matched_pairs(csv_index, nc_index)
    if not matched_pairs:
        raise ValueError(
            "No matched CSV/NetCDF pairs found "
            f"(CSV={len(csv_index)}, NetCDF={len(nc_index)})",
        )

    staging_dir = Path(tempfile.mkdtemp(prefix=".merged-staging-", dir=out_dir.parent))
    try:
        outputs: list[Path] = []
        for station, var, csv_path, nc_path in matched_pairs:
            outputs.append(
                process_matched_pair(station, var, csv_path, nc_path, staging_dir),
            )
        outputs.extend(
            process_csv_only_pairs(
                csv_index,
                nc_index,
                matched_pairs,
                staging_dir,
                station_coordinate_index,
            )
        )
        if not outputs:
            raise RuntimeError("Unification produced no output files")
        source_paths = sorted(
            {Path(path) for path in [*csv_index.values(), *nc_index.values()]}
        )
        manifest = {
            "preprocessing_version": "raw-filter-before-hourly-v2",
            "created_at": datetime.now(UTC).isoformat(),
            "csv_dir": str(csv_dir.resolve()),
            "nc_dir": str(nc_dir.resolve()),
            "sources": [
                {
                    "path": str(path.resolve()),
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in source_paths
            ],
            "outputs": sorted(path.name for path in outputs),
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # A failed run leaves no partial dataset.
        backup_dir = out_dir.with_name(f".{out_dir.name}.previous")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if out_dir.exists():
            out_dir.rename(backup_dir)
        try:
            staging_dir.rename(out_dir)
        except OSError:
            if backup_dir.exists() and not out_dir.exists():
                backup_dir.rename(out_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        return sorted(out_dir.glob("*_merged.nc"))
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for dataset unification."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Unify CSV station dumps with ERA5 NetCDF files"
        " into merged NetCDFs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="YAML configuration file",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help=f"Directory containing station CSV files (default: {CSV_DIR})",
    )
    parser.add_argument(
        "--nc-dir",
        type=Path,
        default=None,
        help=f"Directory containing ERA5 NetCDF files (default: {NC_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Temporary output directory for merged NetCDF files",
    )
    args = parser.parse_args(argv)
    if not args.config.is_file():
        parser.error(f"configuration file not found: {args.config}")
    import yaml  # noqa: PLC0415

    try:
        loaded = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        parser.error(f"cannot load configuration: {exc}")
    if not isinstance(loaded, dict) or not isinstance(loaded.get("paths", {}), dict):
        parser.error("configuration 'paths' section must be a mapping")
    paths = loaded.get("paths", {})
    csv_dir = args.csv_dir or Path(paths.get("raw_csv", CSV_DIR))
    nc_dir = args.nc_dir or Path(paths.get("raw_era5", NC_DIR))
    out_dir = args.out_dir
    outputs = run_unification(csv_dir=csv_dir, nc_dir=nc_dir, out_dir=out_dir)
    logger.info("Published %d unified files", len(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
