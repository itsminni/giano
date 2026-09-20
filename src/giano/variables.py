"""Canonical meteorological variable names."""

VARIABLE_TYPE_MAP: dict[str, int] = {
    "temperature": 0,
    "precipitation": 1,
    "humidity": 2,
    "pressure": 3,
    "wind_speed": 4,
    "wind_direction": 5,
}
VARIABLE_TYPE_NAMES: tuple[str, ...] = tuple(VARIABLE_TYPE_MAP)
