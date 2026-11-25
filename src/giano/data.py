"""Read hourly station data and construct the input grid without changing observations."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .model import VARIABLES
from .physical import PHYSICAL_BOUNDS_BY_VARIABLE, UNITS_BY_VARIABLE
from .windows import LongPanel

ERA5_UNITS = {"t2m": "K", "tp": "m", "sp": "Pa", "u10": "m s-1", "v10": "m s-1"}


def read_csv(path: Path, stations: Path) -> xr.Dataset:
    records = pd.read_csv(path, dtype={"station_id": str})
    catalog = pd.read_csv(stations, dtype={"station_id": str}).set_index("station_id")
    if not catalog.index.is_unique or catalog.index.hasnans:
        raise ValueError("Station catalogue requires unique, nonmissing IDs")
    if records.station_id.isna().any() or not set(records.station_id) <= set(
        catalog.index
    ):
        raise ValueError("Missing or unknown station_id in observations")
    records["time"] = pd.to_datetime(
        records.pop("timestamp"), utc=True, format="ISO8601"
    ).dt.tz_localize(None)
    indexed = records.rename(columns={"station_id": "station"}).set_index(
        ["time", "station"]
    )
    if not indexed.index.is_unique or records["time"].isna().any():
        raise ValueError("Expected unique, nonmissing station timestamps")
    ds = indexed.to_xarray()
    for key in ("latitude", "longitude"):
        ds = ds.assign_coords(
            {key: ("station", catalog.loc[ds.station.values, key].to_numpy())}
        )
    for field in ds.data_vars:
        key = str(field)
        variable = key.removeprefix("aux_")
        if variable in VARIABLES:
            ds[key].attrs["units"] = UNITS_BY_VARIABLE[variable]
        elif key in ERA5_UNITS:
            ds[key].attrs["units"] = ERA5_UNITS[key]
    return ds


def load_input(path: Path, stations: Path | None) -> xr.Dataset:
    if path.suffix.lower() == ".csv":
        if stations is None:
            raise ValueError("CSV observations require a station coordinate catalogue")
        ds = read_csv(path, stations)
    else:
        if stations is not None:
            raise ValueError("NetCDF input carries its own latitude and longitude")
        with xr.open_dataset(path) as source:
            ds = source.load()
    if not any(v in ds for v in VARIABLES):
        raise ValueError("Input has no supported meteorological variable")
    if ds.time.dtype.kind != "M" or ds.station.dims != ("station",):
        raise ValueError("Expected datetime time axis and station identifiers")
    times = ds.time.values.astype("datetime64[ns]")
    if (
        not len(times)
        or np.isnat(times).any()
        or np.any(times != times.astype("datetime64[h]"))
        or np.any(times[1:] <= times[:-1])
    ):
        raise ValueError("Expected ordered, unique whole-hour timestamps")
    ids = [str(s) for s in ds.station.values]
    if not ids or len(set(ids)) != len(ids) or any(not s.strip() for s in ids):
        raise ValueError("Expected unique nonempty station IDs")
    for key, limit in (("latitude", 90), ("longitude", 180)):
        coord = ds[key]
        if (
            coord.dims != ("station",)
            or not np.isfinite(coord).all()
            or np.any(abs(coord.values) > limit)
        ):
            raise ValueError(f"Invalid WGS84 {key}")
    for key, unit in {
        **{v: UNITS_BY_VARIABLE[v] for v in VARIABLES},
        **{"aux_" + v: UNITS_BY_VARIABLE[v] for v in VARIABLES},
        **ERA5_UNITS,
    }.items():
        if key in ds and (
            set(ds[key].dims) != {"time", "station"}
            or ds[key].attrs.get("units") != unit
            or ds[key].dtype.kind not in "iuf"
        ):
            raise ValueError(f"{key} requires numeric time × station data in {unit}")
        if key in ds and np.isinf(ds[key].values).any():
            raise ValueError(f"Replace infinite inputs in {key} with missing values")
    for var in VARIABLES:
        if var in ds:
            a = ds[var].values
            lo, hi = PHYSICAL_BOUNDS_BY_VARIABLE[var]
            if ((a < lo) | (a > hi)).any():
                raise ValueError(f"Check units and missing-value codes for {var}")
    grid = np.arange(
        times[0], times[-1] + np.timedelta64(1, "h"), np.timedelta64(1, "h")
    )
    return (
        ds.assign_coords(station=ids)
        .reindex(time=grid)
        .transpose("time", "station", ...)
    )


def make_panel(ds: xr.Dataset, history_end: int) -> LongPanel:
    shape = (ds.sizes["time"], ds.sizes["station"], len(VARIABLES))
    values = np.full(shape, np.nan, np.float32)
    aux = np.full_like(values, np.nan)
    available = np.zeros(shape, dtype=bool)
    for col, var in enumerate(VARIABLES):
        if var not in ds:
            continue
        values[..., col] = ds[var].values
        if "aux_" + var in ds:
            aux[..., col] = ds["aux_" + var].values
        for node in range(shape[1]):
            rows = np.flatnonzero(np.isfinite(values[:, node, col]))
            if len(rows):
                available[rows[0] : rows[-1] + 1, node, col] = True
    return LongPanel(
        ds.time.values.astype("datetime64[ns]"),
        tuple(ds.station.values.tolist()),
        np.column_stack((ds.latitude.values, ds.longitude.values)).astype(np.float32),
        values,
        aux,
        available,
        history_end,
    )
