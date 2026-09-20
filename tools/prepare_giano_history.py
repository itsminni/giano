"""Export hourly station histories as annual JSON files."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import xarray as xr
from generate_giano_history import verify_completed
from prepare_giano_station_data import safe_path, verify_bundle, write_json

from giano.provenance import file_sha256
from giano.spatiotemporal_dataset import _auxiliary_values
from giano.variables import VARIABLE_TYPE_NAMES

LOGGER = logging.getLogger(__name__)
TIMEZONE = "not established by source"
COUNT_KEYS = (
    "hours",
    "observed_hours",
    "natural_gap_hours",
    "reconstructed_hours",
    "unfilled_hours",
)


def counts(observed: np.ndarray, estimates: np.ndarray) -> dict[str, int]:
    known = int(np.isfinite(observed).sum())
    reconstructed = int(np.isfinite(estimates).sum())
    return dict(
        zip(
            COUNT_KEYS,
            (
                len(observed),
                known,
                len(observed) - known,
                reconstructed,
                len(observed) - known - reconstructed,
            ),
            strict=True,
        )
    )


def add_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {key: sum(row[key] for row in rows) for key in COUNT_KEYS}


def nullable(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def stats(values: np.ndarray, variable: str) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    mean: float | None = None
    if len(finite):
        if variable == "wind_direction":
            vector = np.exp(1j * np.deg2rad(finite)).mean()
            if abs(vector) > 1e-7:
                mean = float(np.rad2deg(np.angle(vector)) % 360)
        else:
            mean = float(finite.mean())
    return {
        "count": len(finite),
        "mean": mean,
        "mean_kind": "circular" if variable == "wind_direction" else "arithmetic",
        "min": float(finite.min()) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
    }


def period_slices(times: np.ndarray, unit: str):
    periods = times.astype(f"datetime64[{unit}]")
    _, starts, lengths = np.unique(periods, return_index=True, return_counts=True)
    for start, length in zip(starts, lengths, strict=True):
        yield str(periods[start]), slice(int(start), int(start + length))


def monthly_overview(
    times: np.ndarray, observed: np.ndarray, estimates: np.ndarray, variable: str
) -> list[dict[str, Any]]:
    return [
        {
            "month": month,
            "start": str(times[span.start]),
            "end": str(times[span.stop - 1]),
            **counts(observed[span], estimates[span]),
            "observed": stats(observed[span], variable),
            "giano_estimates": stats(estimates[span], variable),
        }
        for month, span in period_slices(times, "M")
    ]


def load_history(
    source: Path, reconstructed: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with xr.open_dataset(source, engine="h5netcdf") as original:
        times = np.asarray(original.time.values).astype("datetime64[ns]")
        observed = np.asarray(original.value.values, dtype=np.float64)
        unit = str(original.value.attrs.get("units", ""))
    with xr.open_dataset(reconstructed, engine="h5netcdf") as result:
        filled = np.asarray(result.imputed_value.values, dtype=np.float64)
        support = np.asarray(result.window_prediction_count.values)
        if not np.array_equal(result.time.values, times) or not np.array_equal(
            result.value.values, observed, equal_nan=True
        ):
            raise ValueError("Reconstruction altered source timestamps or observations")
    if observed.ndim != 1 or not len(times) or len(observed) != len(times):
        raise ValueError("Expected a nonempty one-dimensional hourly history")
    if (
        np.isnat(times).any()
        or np.any(np.diff(times) != np.timedelta64(1, "h"))
        or np.any(times != times.astype("datetime64[h]"))
    ):
        raise ValueError("History must have regular, unique, hour-aligned timestamps")
    known = np.isfinite(observed)
    if (
        np.isinf(observed).any()
        or np.isinf(filled).any()
        or not np.array_equal(filled[known], observed[known])
    ):
        raise ValueError("Invalid or changed observed values")
    if np.any(support[known] != 0) or np.any(
        (support > 0) != (~known & np.isfinite(filled))
    ):
        raise ValueError("Every estimate must be supported by a Giano input window")
    estimates = np.where(known, np.nan, filled)
    return times.astype("datetime64[s]"), observed, estimates, unit


def collect_inputs(data: Path, inference: Path) -> list[dict[str, Any]]:
    inputs = []
    for variable in VARIABLE_TYPE_NAMES:
        directory = inference / variable
        completed = verify_completed(directory)
        request = completed["request"]
        manifest = json.loads((directory / f"{variable}_manifest.json").read_text())
        if (
            request["fallback"] != "none"
            or request["seed"] != 42
            or manifest["model_family"] != "imputeformer"
            or manifest["fallback"] != "none"
            or manifest["checkpoint_sha256"] != request["checkpoint_sha256"]
        ):
            raise ValueError(
                "History requires seed-42 Giano with unsupported gaps left empty"
            )
        sources = request["dataset"]["files"]
        actual_sources = {
            p.relative_to(data).as_posix()
            for p in data.glob(f"*/*_{variable}_merged.nc")
        }
        if actual_sources != {row["path"] for row in sources}:
            raise ValueError("History source file membership changed")
        for row in sources:
            source = safe_path(data, row["path"])
            if file_sha256(source) != row["sha256"]:
                raise ValueError(f"Changed history source: {source}")
            reconstructed = directory / f"{source.stem}_imputed.nc"
            if reconstructed.name not in completed["files"]:
                raise ValueError("Missing reconstructed station")
            inputs.append(
                {
                    "station": source.name.split("_", 1)[0],
                    "variable": variable,
                    "source": source,
                    "reconstructed": reconstructed,
                    "source_sha256": row["sha256"],
                    "checkpoint_sha256": request["checkpoint_sha256"],
                }
            )
    return inputs


def load_era5(source: Path, variable: str) -> np.ndarray | None:
    with xr.open_dataset(source, engine="h5netcdf") as dataset:
        return _auxiliary_values(dataset, variable)


def example_era5(
    times: np.ndarray, auxiliary: np.ndarray, timestamps: list[str]
) -> list[float | None]:
    target = np.asarray(timestamps, dtype="datetime64[s]")
    indices = np.searchsorted(times, target)
    if np.any(indices >= len(times)) or not np.array_equal(times[indices], target):
        raise ValueError("Example timestamps do not match the ERA5 source")
    return nullable(auxiliary[indices])


def export_history(
    item: dict[str, Any], destination: Path, example_path: str | None = None
) -> dict[str, Any]:
    station, variable = item["station"], item["variable"]
    times, observed, estimates, unit = load_history(
        item["source"], item["reconstructed"]
    )
    auxiliary = load_era5(item["source"], variable)
    if auxiliary is not None and example_path is not None:
        path = safe_path(destination, example_path)
        example = json.loads(path.read_text())
        example["series"]["era5_land"] = example_era5(
            times, auxiliary, example["series"]["timestamps"]
        )
        write_json(path, example)
    prefix = f"histories/{station}/{variable}"
    chunks = []
    for year, span in period_slices(times, "Y"):
        relative = f"{prefix}/{year}.json"
        payload = {
            "schema_version": 1,
            "station": station,
            "variable": variable,
            "unit": unit,
            "time_start": str(times[span.start]),
            "step_seconds": 3600,
            "length": span.stop - span.start,
            "timestamp_timezone": TIMEZONE,
            "observed": nullable(observed[span]),
            "giano_estimates": nullable(estimates[span]),
        }
        if auxiliary is not None:
            payload["era5_land"] = nullable(auxiliary[span])
        path = safe_path(destination, relative)
        write_json(path, payload)
        chunks.append(
            {
                "year": year,
                "path": relative,
                "start": payload["time_start"],
                "end": str(times[span.stop - 1]),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                **counts(observed[span], estimates[span]),
            }
        )
    index = {
        "schema_version": 1,
        "station": station,
        "variable": variable,
        "unit": unit,
        "model": "Giano",
        "seed": 42,
        "checkpoint_sha256": item["checkpoint_sha256"],
        "source_sha256": item["source_sha256"],
        "timestamp_timezone": TIMEZONE,
        "start": str(times[0]),
        "end": str(times[-1]),
        "step_seconds": 3600,
        "scope": "complete locally available processed hourly record, including all dataset splits",
        "estimate_scope": "natural gaps with local input context; historical reconstruction",
        "ground_truth_in_natural_gaps": None,
        **counts(observed, estimates),
        "chunks": chunks,
        "monthly_overview": monthly_overview(times, observed, estimates, variable),
    }
    write_json(safe_path(destination, f"{prefix}/index.json"), index)
    return {
        "station": station,
        "variable": variable,
        "unit": unit,
        "index": f"{prefix}/index.json",
        "start": index["start"],
        "end": index["end"],
        **counts(observed, estimates),
    }


def verify_histories(bundle: Path, data: Path, inference: Path) -> dict[str, Any]:
    checks: dict[str, Any] = verify_bundle(bundle)
    inputs = collect_inputs(data, inference)
    catalog = json.loads((bundle / "history.json").read_text())
    exported = {(row["station"], row["variable"]): row for row in catalog["series"]}
    expected = {(item["station"], item["variable"]) for item in inputs}
    if set(exported) != expected or len(exported) != len(catalog["series"]):
        raise ValueError("Missing or duplicate historical series")
    all_counts = []
    chunk_count = 0
    for item in inputs:
        key = (item["station"], item["variable"])
        entry = exported[key]
        index = json.loads(safe_path(bundle, entry["index"]).read_text())
        times, observed, estimates, unit = load_history(
            item["source"], item["reconstructed"]
        )
        auxiliary = load_era5(item["source"], item["variable"])
        card = json.loads((bundle / "stations" / f"{key[0]}.json").read_text())
        example_link = card["variables"][key[1]]["example"]
        if example_link and auxiliary is not None:
            example = json.loads(safe_path(bundle, example_link["json"]).read_text())
            if example["series"].get("era5_land") != example_era5(
                times, auxiliary, example["series"]["timestamps"]
            ):
                raise ValueError("Changed example ERA5 values or alignment")
        if (
            index["station"] != key[0]
            or index["variable"] != key[1]
            or index["unit"] != unit
            or index["source_sha256"] != item["source_sha256"]
            or index["checkpoint_sha256"] != item["checkpoint_sha256"]
        ):
            raise ValueError("History provenance or identity mismatch")
        offset = 0
        for descriptor in index["chunks"]:
            path = safe_path(bundle, descriptor["path"])
            chunk = json.loads(path.read_text())
            count = chunk["length"]
            span = slice(offset, offset + count)
            if count <= 0 or count > 8784 or offset + count > len(times):
                raise ValueError("Invalid annual chunk length")
            if (
                file_sha256(path) != descriptor["sha256"]
                or path.stat().st_size != descriptor["bytes"]
            ):
                raise ValueError("Invalid chunk integrity descriptor")
            if (
                chunk["station"] != key[0]
                or chunk["variable"] != key[1]
                or chunk["unit"] != unit
                or chunk["step_seconds"] != 3600
                or chunk["time_start"] != str(times[offset])
                or descriptor["end"] != str(times[offset + count - 1])
            ):
                raise ValueError("Invalid hourly chunk alignment")
            for field, values in (
                ("observed", observed),
                ("giano_estimates", estimates),
            ):
                if not np.array_equal(
                    np.asarray(chunk[field], dtype=float), values[span], equal_nan=True
                ):
                    raise ValueError(f"Changed exported historical values: {field}")
            if auxiliary is not None and (
                "era5_land" not in chunk
                or not np.array_equal(
                    np.asarray(chunk["era5_land"], dtype=float),
                    auxiliary[span],
                    equal_nan=True,
                )
            ):
                raise ValueError("Changed exported ERA5 values or alignment")
            if auxiliary is None and "era5_land" in chunk:
                raise ValueError("ERA5 series has no source")
            if any(
                descriptor[k] != v
                for k, v in counts(observed[span], estimates[span]).items()
            ):
                raise ValueError("Incorrect chunk counts")
            offset += count
            chunk_count += 1
        row_counts = counts(observed, estimates)
        if offset != len(times) or any(
            index[k] != v or entry[k] != v for k, v in row_counts.items()
        ):
            raise ValueError("Truncated history or incorrect counts")
        if index["monthly_overview"] != monthly_overview(
            times, observed, estimates, item["variable"]
        ):
            raise ValueError("Changed monthly overview")
        all_counts.append(row_counts)
    features = json.loads((bundle / "stations.geojson").read_text())["features"]
    linked = set()
    for feature in features:
        card = json.loads(safe_path(bundle, feature["properties"]["card"]).read_text())
        station_keys = {key for key in expected if key[0] == card["id"]}
        if card["has_history"] != bool(station_keys) or feature["properties"][
            "has_history"
        ] != bool(station_keys):
            raise ValueError("Invalid map/card history availability")
        for variable, entry in card["variables"].items():
            key = (card["id"], variable)
            if key in expected:
                if entry["history"] != exported[key]["index"]:
                    raise ValueError("Invalid card/history link")
                linked.add(key)
            elif entry["history"] is not None:
                raise ValueError("Unexpected card history")
    totals = add_counts(all_counts)
    if linked != expected or any(catalog["counts"][k] != v for k, v in totals.items()):
        raise ValueError("Historical catalog counts or links disagree")
    checks.update(
        {
            "series": len(inputs),
            "stations_with_history": len({k[0] for k in expected}),
            "annual_chunks": chunk_count,
            **totals,
            "exact_source_values_verified": True,
        }
    )
    return checks


def build(source: Path, destination: Path, data: Path, inference: Path) -> None:
    verify_bundle(source)
    inputs = collect_inputs(data, inference)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("README.md"))
    manifest = json.loads((destination / "manifest.json").read_text())
    features = json.loads((destination / "stations.geojson").read_text())["features"]
    cards = {
        feature["id"]: json.loads(
            safe_path(destination, feature["properties"]["card"]).read_text()
        )
        for feature in features
    }
    series = []
    for number, item in enumerate(inputs, start=1):
        card = cards[item["station"]]
        if item["variable"] not in card["variables"]:
            raise ValueError("Historical variable missing from station card")
        example = card["variables"][item["variable"]]["example"]
        entry = export_history(item, destination, example["json"] if example else None)
        card["variables"][entry["variable"]]["history"] = entry["index"]
        series.append(entry)
        if number == 1 or number % 25 == 0 or number == len(inputs):
            LOGGER.info(
                "Exported %d/%d complete station-variable histories",
                number,
                len(inputs),
            )
    for feature in features:
        card = cards[feature["id"]]
        for item in card["variables"].values():
            item.setdefault("history", None)
        card["has_history"] = any(
            item["history"] is not None for item in card["variables"].values()
        )
        feature["properties"]["has_history"] = card["has_history"]
        write_json(safe_path(destination, feature["properties"]["card"]), card)
    write_json(
        destination / "stations.geojson",
        {"type": "FeatureCollection", "features": features},
    )
    totals = {
        "series": len(series),
        "stations": sum(card["has_history"] for card in cards.values()),
        **add_counts(series),
    }
    write_json(
        destination / "history.json",
        {
            "schema_version": 1,
            "model": "Giano",
            "seed": 42,
            "scope": "complete local processed hourly archive",
            "start": min(row["start"] for row in series),
            "end": max(row["end"] for row in series),
            "timestamp_timezone": TIMEZONE,
            "counts": totals,
            "series": series,
        },
    )
    content = json.loads((destination / "content.json").read_text())
    content["en"]["history_description"] = (
        "Browse the available station histories, with hourly observations and Giano reconstructions."
    )
    content["it"]["history_description"] = (
        "Consultare le serie storiche disponibili per ogni stazione, con osservazioni orarie e ricostruzioni di Giano."
    )
    write_json(destination / "content.json", content)
    manifest.update(
        {
            "built_at": datetime.now(UTC).isoformat(),
            "history_schema_version": 1,
            "history_counts": totals,
            "presentation_source_manifest_sha256": file_sha256(
                source / "manifest.json"
            ),
            "history_builder_sha256": file_sha256(Path(__file__)),
        }
    )
    manifest["files"] = {
        p.relative_to(destination).as_posix(): file_sha256(p)
        for p in sorted(destination.rglob("*"))
        if p.is_file() and p != destination / "manifest.json"
    }
    write_json(destination / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/website/giano-stations")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/2-processed-v2"))
    parser.add_argument(
        "--inference", type=Path, default=Path("artifacts/imputed/giano_history")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/website/giano-history")
    )
    parser.add_argument(
        "--archive", type=Path, default=Path("artifacts/website/giano-history.zip")
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.verify_only:
        print(
            json.dumps(
                verify_histories(args.output, args.data_dir, args.inference), indent=2
            )
        )
        return
    if args.output.exists() or args.archive.exists():
        raise FileExistsError(
            "Preserve existing output/archive, choose new destinations"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".giano-history-", dir=args.output.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        build(args.source, staging, args.data_dir, args.inference)
        LOGGER.info("Verifying all exported hourly values against the original records")
        checks = verify_histories(staging, args.data_dir, args.inference)
        staging.rename(args.output)
    with ZipFile(args.archive, "x", compression=ZIP_DEFLATED) as archive:
        for path in sorted(args.output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(args.output.parent))
    with ZipFile(args.archive) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity failure")
    report = {
        "output": str(args.output.resolve()),
        "archive": str(args.archive.resolve()),
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": file_sha256(args.archive),
        **checks,
    }
    write_json(args.output.parent / f"{args.output.name}-verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
