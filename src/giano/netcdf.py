"""NetCDF and xarray helpers used across Giano."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import xarray as xr


def find_time_coord(ds: xr.Dataset) -> str | None:
    """Return name of the time-like coordinate in an xarray Dataset, or None.

    Checks a list of common coordinate names first, then falls back to
    inspecting dtype kind == 'M' (datetime64).
    """
    candidates = ["time", "valid_time", "times", "datetime", "date", "time_utc"]
    for c in candidates:
        if c in ds.coords:
            return c
    for coord_name, coord in ds.coords.items():
        dtype = getattr(coord, "dtype", None)
        kind = getattr(dtype, "kind", None)
        if kind == "M":
            return str(coord_name)
    return None


def available_netcdf_engines() -> list[str]:
    """Return installed xarray NetCDF engines in preferred order."""
    engine_modules = {
        "h5netcdf": "h5netcdf",
        "netcdf4": "netCDF4",
        "scipy": "scipy",
    }
    installed = []
    for engine, module_name in engine_modules.items():
        if importlib.util.find_spec(module_name) is not None:
            installed.append(engine)
    return installed


def open_dataset_robust(path: str | Path) -> xr.Dataset:
    """Open a NetCDF file using the first working xarray backend.

    Raises a clear RuntimeError when no usable backend is available.
    """
    nc_path = Path(path)
    engines = available_netcdf_engines()
    if not engines:
        msg = (
            "No NetCDF backend found for xarray. Restore the locked project "
            "environment with 'uv sync --locked'."
        )
        raise RuntimeError(msg)

    backend_errors: list[str] = []
    for engine in engines:
        try:
            return xr.open_dataset(nc_path, engine=engine)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:  # noqa: PERF203
            backend_errors.append(f"{engine}: {exc}")

    joined_errors = "; ".join(backend_errors)
    msg = (
        f"Could not open NetCDF file '{nc_path}' with available engines "
        f"{engines}. Errors: {joined_errors}"
    )
    raise RuntimeError(msg)
