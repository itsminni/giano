"""Explicit, offline CSV adapter for hourly and regular sub-hourly stations."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from giano.data_processing.adapters.base import DatasetAdapter
from giano.data_processing.unify_dataset import (
    _resample_station_frame,
    _station_dataset,
)
from giano.variables import VARIABLE_TYPE_NAMES

# source units -> multiplier, offset. Applied only after source missing flags.
UNIT_CONVERSIONS = {
    "temperature": {"degC": (1.0, 0.0), "K": (1.0, -273.15), "degF": (5 / 9, -160 / 9)},
    "precipitation": {"mm": (1.0, 0.0), "m": (1000.0, 0.0)},
    "humidity": {"%": (1.0, 0.0), "fraction": (100.0, 0.0)},
    "pressure": {"hPa": (1.0, 0.0), "Pa": (0.01, 0.0)},
    "wind_speed": {
        "m s-1": (1.0, 0.0),
        "km h-1": (1 / 3.6, 0.0),
        "knots": (1852 / 3600, 0.0),
    },
    "wind_direction": {"degree": (1.0, 0.0), "radian": (180 / np.pi, 0.0)},
}
VARIABLE_KEYS = {
    "column",
    "unit",
    "missing_values",
    "quality_column",
    "invalid_quality_values",
    "precipitation_kind",
    "timestamp_position",
}


def station_key(identifier: str) -> str:
    """Reversible, collision-free, path-safe IDs accepted by all loaders."""
    if (
        not identifier
        or identifier != identifier.strip()
        or len(identifier.encode("utf-8")) > 80
    ):
        raise ValueError(
            "Station IDs must be nonempty, trimmed and at most 80 UTF-8 bytes"
        )
    return "S" + identifier.encode("utf-8").hex()


def _tokens(raw: Any, name: str) -> list[str]:
    if not isinstance(raw, list) or any(
        not isinstance(item, (str, int, float)) or isinstance(item, bool)
        for item in raw
    ):
        raise ValueError(f"{name} must be a list of strings/numbers")
    return [str(item).strip() for item in raw]


def _numeric(series: pd.Series, missing: list[str], name: str) -> pd.Series:
    """Reject unexplained strings."""
    clean = series.astype(str).str.strip()
    absent = clean.isin(["", "NaN", "nan", "NA", "null", *missing])
    result = pd.to_numeric(clean.where(~absent), errors="coerce")
    # A numeric sentinel is independent of spelling: -999 and -999.0 agree.
    numeric_missing = pd.to_numeric(
        pd.Series(missing, dtype=str), errors="coerce"
    ).dropna()
    absent |= result.isin(numeric_missing)
    if (result.isna() & ~absent).any():
        raise ValueError(
            f"Non-numeric {name}: declare provider missing_values explicitly"
        )
    return result.where(~absent)


def _utc_times(series: pd.Series, timezone: str) -> pd.DatetimeIndex:
    """Never guess DST or mix local timestamps with explicit UTC offsets."""
    try:
        times = pd.DatetimeIndex(
            pd.to_datetime(series, format="ISO8601", errors="raise")
        )
        if times.tz is None:
            times = times.tz_localize(
                ZoneInfo(timezone), ambiguous="raise", nonexistent="raise"
            )
        elif timezone != "UTC":
            raise ValueError(
                "Offset-aware timestamps require timezone: UTC. Do not mix conventions"
            )
        return times.tz_convert("UTC").tz_localize(None).as_unit("ns")
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(
            f"Invalid/ambiguous timestamps for timezone {timezone}: {exc}"
        ) from exc


class CsvDatasetAdapter(DatasetAdapter):
    """One wide CSV, one station catalogue, explicit per-variable semantics."""

    def __init__(self, config: dict[str, Any], directory: Path) -> None:
        self.config = config
        self.observations = directory / str(config["observations"])
        self.stations = directory / str(config["stations"])
        self.time_column = str(config.get("timestamp_column", "timestamp"))
        self.station_column = str(config.get("station_column", "station_id"))
        self.timezone = str(config["timezone"])
        ZoneInfo(self.timezone)
        self.cadence = config["cadence_minutes"]
        if (
            isinstance(self.cadence, bool)
            or not isinstance(self.cadence, int)
            or self.cadence <= 0
            or 60 % self.cadence
        ):
            raise ValueError(
                "cadence_minutes must be a positive integer divisor of 60 (no daily upsampling)"
            )
        self.variables = config["variables"]
        if not isinstance(self.variables, dict) or not self.variables:
            raise ValueError("variables must be a nonempty mapping")
        for variable, spec in self.variables.items():
            if variable not in VARIABLE_TYPE_NAMES:
                raise ValueError(f"Unsupported target variable: {variable}")
            if not isinstance(spec, dict) or set(spec) - VARIABLE_KEYS:
                raise ValueError(f"Invalid/unknown configuration keys for {variable}")
            if not isinstance(spec.get("column"), str) or not spec["column"]:
                raise ValueError(f"{variable} requires a source column")
            if spec.get("unit") not in UNIT_CONVERSIONS[variable]:
                raise ValueError(
                    f"Unsupported unit for {variable}; choose {list(UNIT_CONVERSIONS[variable])}"
                )
            _tokens(spec.get("missing_values", []), "missing_values")
            _tokens(spec.get("invalid_quality_values", []), "invalid_quality_values")
            if ("quality_column" in spec) != ("invalid_quality_values" in spec):
                raise ValueError(
                    "quality_column and invalid_quality_values must be supplied together"
                )
            if variable == "precipitation":
                if spec.get("precipitation_kind") != "interval_total" or spec.get(
                    "timestamp_position"
                ) not in {"start", "end"}:
                    raise ValueError(
                        "Precipitation requires interval_total and timestamp_position: start/end; rates/counters are not accepted"
                    )
            elif {"precipitation_kind", "timestamp_position"} & set(spec):
                raise ValueError("Interval semantics apply only to precipitation")

    def source_paths(self) -> list[Path]:
        return [self.observations, self.stations]

    def write(self, destination: Path) -> list[Path]:
        separator = self.config.get("delimiter", ",")
        if not isinstance(separator, str) or len(separator) != 1:
            raise ValueError("delimiter must be one character")
        catalogue = pd.read_csv(self.stations, dtype=str, keep_default_na=False)
        if not {"station_id", "latitude", "longitude"}.issubset(catalogue):
            raise ValueError(
                "Station catalogue requires station_id,latitude,longitude (WGS84 degrees)"
            )
        if catalogue.empty or catalogue["station_id"].duplicated().any():
            raise ValueError("Station catalogue must be nonempty with unique IDs")
        for identifier in catalogue["station_id"]:
            station_key(identifier)
        catalogue = catalogue.set_index("station_id")
        for column, limit in (("latitude", 90), ("longitude", 180)):
            numeric = pd.to_numeric(catalogue[column], errors="coerce")
            if not (np.isfinite(numeric) & numeric.between(-limit, limit)).all():
                raise ValueError(
                    f"Invalid station {column}; supply finite WGS84 degrees"
                )
            catalogue[column] = numeric
        observations = pd.read_csv(
            self.observations, sep=separator, dtype=str, keep_default_na=False
        )
        required = {self.time_column, self.station_column}
        for spec in self.variables.values():
            required.add(spec["column"])
            if "quality_column" in spec:
                required.add(spec["quality_column"])
        if observations.empty or not required.issubset(observations):
            raise ValueError(
                f"Nonempty observation CSV requires columns {sorted(required)}"
            )
        unknown = set(observations[self.station_column]) - set(catalogue.index)
        if unknown:
            raise ValueError(f"Missing station coordinates for {sorted(unknown)}")
        outputs = []
        group = destination / "all"
        group.mkdir(parents=True, exist_ok=True)
        for identifier, raw in observations.groupby(self.station_column, sort=True):
            station = str(identifier)
            times = _utc_times(raw[self.time_column], self.timezone)
            if times.hasnans or times.has_duplicates:
                raise ValueError(
                    f"Duplicate or missing timestamps for station {station}"
                )
            if np.any(
                times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
                % (self.cadence * 60 * 10**9)
            ):
                raise ValueError(
                    f"Timestamps for {station} are off the declared UTC cadence grid"
                )
            if (times.max() - times.min()) / pd.Timedelta(hours=1) > 2_000_000:
                raise ValueError("Implausibly long time axis, check timestamp parsing")
            for variable, spec in self.variables.items():
                values = _numeric(
                    raw[spec["column"]],
                    _tokens(spec.get("missing_values", []), "missing_values"),
                    variable,
                )
                if "quality_column" in spec:
                    # Provider-specific tokens only.
                    quality = raw[spec["quality_column"]].astype(str).str.strip()
                    invalid = _tokens(
                        spec["invalid_quality_values"], "invalid_quality_values"
                    )
                    invalid_numeric = pd.to_numeric(
                        pd.Series(invalid, dtype=str), errors="coerce"
                    ).dropna()
                    excluded = (
                        quality.eq("")
                        | quality.isin(invalid)
                        | pd.to_numeric(quality, errors="coerce").isin(invalid_numeric)
                    )
                    values = values.where(~excluded)
                multiplier, offset = UNIT_CONVERSIONS[variable][spec["unit"]]
                values = values * multiplier + offset
                index = times
                if variable == "precipitation" and spec["timestamp_position"] == "end":
                    index = times - pd.Timedelta(minutes=self.cadence)
                frame = pd.DataFrame(
                    {"value": values.to_numpy()}, index=index.rename("time")
                ).sort_index()
                hourly = _resample_station_frame(frame, variable)
                if variable == "precipitation":
                    hourly["value"] = hourly["value"].where(
                        hourly["valid_sample_count"] == 60 // self.cadence
                    )
                ds = _station_dataset(hourly, variable)
                ds.attrs.update(
                    {
                        "preprocessing_version": "generic-csv-hourly-v1",
                        "dataset_id": self.config["dataset_id"],
                        "station_id": station_key(station),
                        "original_station_id": station,
                        "station_latitude": float(
                            str(catalogue.loc[station, "latitude"])
                        ),
                        "station_longitude": float(
                            str(catalogue.loc[station, "longitude"])
                        ),
                        "coordinate_source": "user_station_catalogue_wgs84",
                        "timestamp_convention": "UTC; timezone-naive NetCDF encoding",
                        "source_timezone": self.timezone,
                        "source_unit": spec["unit"],
                        "source_cadence_minutes": self.cadence,
                        "hourly_coverage_basis": "declared_cadence; precipitation requires every expected sample; scalars use valid samples",
                        "source_timestamp_position": spec.get(
                            "timestamp_position", "observation_time"
                        ),
                        "auxiliary_source": "none; no auxiliary observations fabricated",
                    }
                )
                if variable == "precipitation":
                    ds["value"].attrs["aggregation"] = (
                        "sum_all_expected_samples_valid; left_labelled_interval"
                    )
                path = group / f"{station_key(station)}_{variable}_merged.nc"
                ds.to_netcdf(path, engine="h5netcdf")
                ds.close()
                outputs.append(path)
        return outputs
