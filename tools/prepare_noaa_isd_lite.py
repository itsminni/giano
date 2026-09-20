"""Convert local NOAA ISD-Lite temperature and wind records to CSV."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from giano.provenance import file_sha256

BASE_URL = "https://www.ncei.noaa.gov/pub/data/noaa/"
COLUMNS = [
    "year",
    "month",
    "day",
    "hour",
    "temperature",
    "dewpoint",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
    "cloud",
    "rain_1h",
    "rain_6h",
]


def prepare(source_dir: Path, output_dir: Path, year: int, stations: list[str]) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output already exists: {output_dir}")
    if (
        not 1901 <= year <= 2100
        or not stations
        or len(set(stations)) != len(stations)
        or any(not re.fullmatch(r"\d{6}-\d{5}", station) for station in stations)
    ):
        raise ValueError("Use one year and unique USAF-WBAN station identifiers")
    metadata_path = source_dir / "isd-history.csv"
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    frames = []
    catalogue = []
    sources = [
        {
            "file": metadata_path.name,
            "url": BASE_URL + metadata_path.name,
            "sha256": file_sha256(metadata_path),
        }
    ]
    for station in stations:
        usaf, wban = station.split("-")
        entry = metadata.loc[(metadata["USAF"] == usaf) & (metadata["WBAN"] == wban)]
        if len(entry) != 1:
            raise ValueError(f"Missing/ambiguous station metadata: {station}")
        record = entry.iloc[0]
        catalogue.append(
            {
                "station_id": station,
                "latitude": float(record["LAT"]),
                "longitude": float(record["LON"]),
                "name": record["STATION NAME"],
            }
        )
        path = source_dir / f"{station}-{year}.gz"
        raw = pd.read_csv(path, sep=r"\s+", header=None, compression="gzip")
        if raw.shape[1] != len(COLUMNS) or raw.empty:
            raise ValueError(f"Invalid ISD-Lite shape in {path}")
        raw.columns = COLUMNS
        if not raw["year"].eq(year).all():
            raise ValueError(f"Year mismatch in {path}")
        timestamp = pd.to_datetime(raw[["year", "month", "day", "hour"]], utc=True)
        measurements = (
            raw[["temperature", "wind_speed", "wind_direction"]]
            .replace(-9999, np.nan)
            .astype(float)
        )
        measurements["temperature"] /= 10
        measurements["wind_speed"] /= 10
        # NOAA direction 0 means calm; 360 means north.
        measurements.loc[
            (measurements["wind_direction"] == 0) | (measurements["wind_speed"] == 0),
            "wind_direction",
        ] = np.nan
        measurements.insert(0, "timestamp", timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        measurements.insert(0, "station_id", station)
        frames.append(measurements)
        sources.append(
            {
                "file": path.name,
                "url": f"{BASE_URL}isd-lite/{year}/{path.name}",
                "sha256": file_sha256(path),
            }
        )
    config = {
        "adapter": "csv",
        "dataset_id": f"noaa-isd-lite-{year}",
        "observations": "observations.csv",
        "stations": "stations.csv",
        "timezone": "UTC",
        "cadence_minutes": 60,
        "variables": {
            "temperature": {"column": "temperature", "unit": "degC"},
            "wind_speed": {"column": "wind_speed", "unit": "m s-1"},
            "wind_direction": {"column": "wind_direction", "unit": "degree"},
        },
    }
    provenance = {
        "dataset": config["dataset_id"],
        "sources": sources,
        "stations": catalogue,
        "format_documentation": BASE_URL + "isd-lite/isd-lite-format.txt",
        "limits": "Nominal hourly observations; no quality flags. Missing code -9999 and calm-wind direction are masked. Temperature and wind only.",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".noaa-example-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / "csv"
        staging.mkdir()
        pd.concat(frames, ignore_index=True).to_csv(
            staging / "observations.csv", index=False
        )
        pd.DataFrame(catalogue).to_csv(staging / "stations.csv", index=False)
        (staging / "import.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        provenance["outputs"] = {
            path.name: file_sha256(path) for path in sorted(staging.iterdir())
        }
        (staging / "source_manifest.json").write_text(
            json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8"
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Output appeared during preparation: {output_dir}")
        staging.rename(output_dir)
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--stations", nargs="+", required=True)
    args = parser.parse_args(argv)
    result = prepare(args.source_dir, args.output_dir, args.year, args.stations)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
