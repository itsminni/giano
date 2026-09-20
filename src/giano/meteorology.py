"""Domain-specific meteorological helpers shared by Giano."""

from __future__ import annotations

import numpy as np

# Conservative physical ranges that fit the Trentino station domain while
# still filtering obvious sentinels / corrupted values.
PHYSICAL_BOUNDS_BY_VARIABLE: dict[str, tuple[float, float]] = {
    "temperature": (-80.0, 60.0),
    "dewpoint": (-90.0, 50.0),
    "precipitation": (0.0, 300.0),
    "humidity": (0.0, 100.0),
    "pressure": (550.0, 1100.0),
    "wind_speed": (0.0, 75.0),
    "wind_direction": (0.0, 360.0),
}

# Meteotrentino quality codes 151 and 255 explicitly mean missing data. They
# can accompany plausible numeric placeholders (for example 0 °C or 800 hPa),
# so physical-range checks alone cannot identify them.
MISSING_QUALITY_CODES = frozenset({151.0, 255.0})

UNITS_BY_VARIABLE = {
    "temperature": "degC",
    "dewpoint": "degC",
    "precipitation": "mm",
    "humidity": "%",
    "pressure": "hPa",
    "wind_speed": "m s-1",
    "wind_direction": "degree",
}


def normalize_era5_units(
    values: np.ndarray,
    variable_name: str | None,
    attrs: dict[str, object] | None,
) -> np.ndarray:
    """Convert common ERA5 units into the station-data convention."""
    normalized = np.asarray(values, dtype=float)
    normalized_name = str(variable_name or "").strip().lower()
    raw_units = ""
    if attrs:
        raw_units = str(attrs.get("units") or attrs.get("GRIB_units") or "").strip()
    units = raw_units.lower()

    if normalized_name in {"temperature", "dewpoint"} and units in {"k", "kelvin"}:
        return normalized - 273.15
    if normalized_name == "pressure" and units in {"pa", "pascal", "pascals"}:
        return normalized / 100.0
    if normalized_name == "precipitation" and units in {"m", "meter", "metre"}:
        return normalized * 1000.0

    return normalized


def apply_physical_bounds(
    values: np.ndarray,
    variable_name: str | None,
    quality_values: np.ndarray | None = None,
) -> np.ndarray:
    """Mask physically invalid values and explicit missing-data quality codes.

    Meteotrentino defines quality codes 151 and 255 as missing data. Other codes
    describe good, estimated, interpolated, uncertain, or not-yet-validated
    observations and remain available as metadata instead of being collapsed
    into a binary validity judgement here.
    """
    masked = np.asarray(values, dtype=float).copy()
    normalized_name = str(variable_name or "").strip().lower()
    bounds = PHYSICAL_BOUNDS_BY_VARIABLE.get(normalized_name)

    invalid_mask = ~np.isfinite(masked)
    if bounds is not None:
        lower_bound, upper_bound = bounds
        invalid_mask |= masked < lower_bound
        invalid_mask |= masked > upper_bound

    if quality_values is not None:
        quality = np.asarray(quality_values, dtype=float)
        try:
            quality = np.broadcast_to(quality, masked.shape)
        except ValueError as exc:
            raise ValueError(
                "quality_values cannot be broadcast to the measurement shape",
            ) from exc
        invalid_mask |= np.isin(quality, tuple(MISSING_QUALITY_CODES))

    if invalid_mask.any():
        masked[invalid_mask] = np.nan
    return masked


def derive_wind_speed_from_uv(
    u_component: np.ndarray, v_component: np.ndarray
) -> np.ndarray:
    """Return wind-speed magnitude from ERA5 u/v components."""
    u = np.asarray(u_component, dtype=float)
    v = np.asarray(v_component, dtype=float)
    return np.sqrt(u**2 + v**2)


def derive_wind_direction_from_uv(
    u_component: np.ndarray,
    v_component: np.ndarray,
) -> np.ndarray:
    """Return meteorological wind direction in degrees from ERA5 u/v components."""
    u = np.asarray(u_component, dtype=float)
    v = np.asarray(v_component, dtype=float)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return direction
