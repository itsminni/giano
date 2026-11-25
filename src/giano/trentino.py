"""Read the original Meteotrentino Drive layout on a bounded source-hour grid."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr

from .data import ERA5_UNITS
from .model import VARIABLES
from .physical import UNITS_BY_VARIABLE
from .windows import LongPanel

PROTOCOL = Path(__file__).with_name("protocol")


def checked_path(root: Path, relative: str, expected: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Supply the Drive data directory containing 1-raw and 2-processed-v2."
        )
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != expected:
            raise ValueError(f"Data differs from the official station file: {path}")
    return path


def humidity_reference_values(
    data: Path, times: np.ndarray, stations: tuple[str, ...]
) -> np.ndarray:
    """Humidity eligible for reference fitting, on the requested station-hour grid."""
    result = np.full((len(times), len(stations)), np.nan, np.float32)
    sources = json.loads((PROTOCOL / "sources.json").read_text())["observations"]
    for name, checksum in sources.items():
        if not name.endswith("_humidity_merged.nc"):
            continue
        site = Path(name).name.split("_")[0]
        if site not in stations:
            continue
        with xr.open_dataset(checked_path(data, name, checksum)) as ds:
            part = ds.sel(time=slice(times[0], times[-1]))
            q, rh = part.quality.values, part.value.values
            use = (
                np.isfinite(q) & ~np.isin(q, (140, 151, 255)) & (rh >= 0) & (rh <= 100)
            )
            result[times.searchsorted(part.time.values[use]), stations.index(site)] = (
                rh[use]
            )
    return result


def _coordinates(ds: xr.Dataset) -> tuple[float, float]:
    def first(names: tuple[str, ...]) -> float | None:
        for name in names:
            if name in ds.coords:
                value = np.asarray(ds.coords[name].values, dtype=float).reshape(-1)
                if value.size and np.isfinite(value[0]):
                    return float(value[0])
            if name in ds.attrs:
                try:
                    attr_value = float(ds.attrs[name])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(attr_value):
                    return attr_value
        return None

    latitude = first(("latitude", "lat", "station_latitude"))
    longitude = first(("longitude", "lon", "station_longitude"))
    if latitude is None or longitude is None:
        raise ValueError("Processed station file lacks latitude/longitude")
    return float(np.clip(latitude, -90, 90)), float(np.clip(longitude, -180, 180))


def read_network(
    root: Path,
    start: str,
    stop: str,
    train_end: str,
    *,
    preserve_precision: bool = False,
) -> tuple[LongPanel, np.ndarray]:
    """The split folder names identify station groups. Bound measurements by date."""
    source = json.loads((PROTOCOL / "sources.json").read_text())
    stations = tuple(source["stations"])
    bounds = np.array([start, stop], dtype="datetime64[ns]")
    if (
        np.isnat(bounds).any()
        or bounds[0] >= bounds[1]
        or (bounds != bounds.astype("datetime64[h]")).any()
    ):
        raise ValueError("Use an increasing interval with whole-hour boundaries")
    times = np.arange(bounds[0], bounds[1], np.timedelta64(1, "h"))
    shape = (len(times), len(stations), len(VARIABLES))
    values = np.full(shape, np.nan, dtype=float if preserve_precision else np.float32)
    auxiliary = np.full_like(values, np.nan)
    available = np.zeros(shape, dtype=bool)
    coords = np.empty((len(stations), 2), dtype=np.float32)
    for relative, sha in source["observations"].items():
        path = checked_path(root, relative, sha)
        station, field = path.stem.removesuffix("_merged").split("_", 1)
        node, col = stations.index(station), VARIABLES.index(field)
        with xr.open_dataset(path) as ds:
            axis = ds.time.values
            available[:, node, col] = (times >= axis[0]) & (times <= axis[-1])
            coords[node] = _coordinates(ds)
            part = ds.sel(time=slice(times[0], times[-1]))
            rows = times.searchsorted(part.time.values)
            values[rows, node, col] = part.value.values
            if field in part:
                auxiliary[rows, node, col] = part[field].values
            elif field.startswith("wind_") and {"wind_u", "wind_v"} <= set(part):
                u = part.wind_u.values.astype(float)
                v = part.wind_v.values.astype(float)
                auxiliary[rows, node, col] = (
                    np.sqrt(u**2 + v**2)
                    if field == "wind_speed"
                    else (270 - np.degrees(np.arctan2(v, u))) % 360
                )
    surface = np.full(shape[:2], np.nan, np.float32)
    for station, entry in source["surface_pressure"].items():
        node = stations.index(station)
        with xr.open_dataset(checked_path(root, entry["file"], entry["sha256"])) as ds:
            field = ds[entry["field"]]
            axis_name = next(k for k in ("time", "valid_time") if k in field.dims)
            bounded = field.sel({axis_name: slice(times[0], times[-1])})
            rows = times.searchsorted(bounded[axis_name].values)
            surface[rows, node] = bounded.values.astype(float).reshape(len(rows)) / 100
    surface[~available[..., 2]] = np.nan
    return LongPanel(
        times,
        stations,
        coords,
        values,
        auxiliary,
        available,
        int(times.searchsorted(np.datetime64(train_end))),
    ), surface


def weather_fields(data, panel, surface, spec, *, start=0):
    times = panel.times[start:]
    ds = xr.Dataset()
    for name, unit in ERA5_UNITS.items():
        ds[name] = (
            ("time", "station"),
            np.full(panel.values.shape[:2], np.nan),
            {"units": unit},
        )
    ds["sp"].values[start:] = surface.astype(float) * 100
    for relative, checksum in spec["weather_files"].items():
        site = Path(relative).stem.split("_")[-1]
        node = panel.station_ids.index(site)
        with xr.open_dataset(checked_path(data, relative, checksum)) as raw:
            part = raw.sel(valid_time=slice(times[0], times[-1]))
            indices = times.searchsorted(part.valid_time.values) + start
            for name in set(ds.data_vars) & set(part.data_vars):
                ds[name].values[indices, node] = part[name].values
    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Convert the Drive station files to a single hourly NetCDF for Giano."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or np.datetime64(args.start) >= np.datetime64(
        args.end_exclusive
    ):
        parser.error("Choose a new output file and a nonempty date range")
    panel, surface = read_network(
        args.data, args.start, args.end_exclusive, args.start, preserve_precision=True
    )
    spec = json.loads((PROTOCOL / "historical.json").read_text())
    ds = weather_fields(args.data, panel, surface, spec)
    ds = ds.assign_coords(
        time=panel.times,
        station=list(panel.station_ids),
        latitude=("station", panel.coordinates[:, 0]),
        longitude=("station", panel.coordinates[:, 1]),
    )
    for j, v in enumerate(VARIABLES):
        unit = {"units": UNITS_BY_VARIABLE[v]}
        ds[v] = (("time", "station"), panel.values[..., j], unit)
        ds["aux_" + v] = (("time", "station"), panel.auxiliary[..., j], unit)
    ds["humidity_reference_valid"] = (
        ("time", "station"),
        np.isfinite(
            humidity_reference_values(args.data, panel.times, panel.station_ids)
        ).astype(np.uint8),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        file = Path(tmp) / "network.nc"
        ds.to_netcdf(
            file,
            engine="h5netcdf",
            encoding={k: {"zlib": True, "complevel": 1} for k in ds.data_vars},
        )
        os.link(file, args.output)
    print(f"{len(panel.times)} hours, {len(panel.station_ids)} stations: {args.output}")
