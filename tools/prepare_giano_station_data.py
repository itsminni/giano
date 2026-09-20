"""Export Giano-only station cards, metrics and plots."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from giano.provenance import file_sha256

DEFAULT_SOURCE = Path("artifacts/website/corrected_v2_release")
DEFAULT_OUTPUT = Path("artifacts/website/giano-stations")
FORBIDDEN = re.compile(
    r"bilstm|\blstm\b|interpolat|baseline|legacy|\bfair\b|comparison|comparativ|"
    r"conclusion|conclusioni|improvement|miglioramento|superiority|\bversus\b|\bSotA\b",
    re.IGNORECASE,
)
SERIES_FIELDS = (
    "timestamps",
    "timestamp_timezone",
    "ground_truth",
    "observed_with_mask",
    "reconstruction",
    "synthetic_mask",
    "ground_truth_unavailable",
)
LEGEND = ("Ground truth", "Visible observations", "Giano reconstruction")


def pick(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: value[key] for key in keys if key in value}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def safe_path(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError(f"Asset path escapes bundle: {relative}")
    return path


def project_results(source: dict[str, Any]) -> dict[str, Any]:
    variables = {}
    for variable, value in source["variables"].items():
        variables[variable] = {
            "unit": value["unit"],
            "metrics": pick(
                value["scores"]["giano/imputeformer"],
                (
                    "mae",
                    "mean_case_seed_rmse",
                    "case_seed_groups",
                ),
            ),
        }
    return {
        "schema_version": 1,
        "model": "Giano",
        "split": source["split"],
        "seeds": source["seeds"],
        "aggregation": "equal weights across 13 validation cases and five seeds",
        "metric_scope": "artificially masked observations",
        "variables": variables,
    }


def project_metrics(source: dict[str, Any] | None) -> dict[str, Any] | None:
    if source is None:
        return None
    result = pick(
        source,
        (
            "split",
            "aggregation",
            "available_case_seed_groups",
            "expected_case_seed_groups",
            "complete_case_seed_coverage",
            "hidden_evaluations_including_repeats",
        ),
    )
    result["metrics"] = pick(source["scores"]["giano"], ("mae", "mean_case_seed_rmse"))
    result["case_seed_metrics"] = [
        {
            **pick(row, ("seed", "mask_type", "mask_parameter", "n_hidden")),
            "mae": row["giano_mae"],
            "rmse": row["giano_rmse"],
        }
        for row in source["case_seed_scores"]
    ]
    return result


def project_example(source: dict[str, Any]) -> dict[str, Any]:
    result = pick(
        source,
        (
            "station",
            "variable",
            "unit",
            "split",
            "seed",
            "selection_seed",
            "sample_index",
            "start_index",
            "illustration_only",
            "included_in_headline_scores",
            "checkpoint_sha256",
        ),
    )
    result.update(
        {
            "schema_version": 1,
            "model": "Giano",
            "selection": "seeded random eligible validation window",
            "case": pick(source["case"], ("mask_type", "requested_length_hours")),
            "metrics": {
                "scope": "this synthetic mask only",
                "hidden_points": source["metrics"]["hidden_points"],
                "mae": source["metrics"]["giano_mae"],
                "rmse": source["metrics"]["giano_rmse"],
            },
            "series": pick(source["series"], SERIES_FIELDS),
        }
    )
    return result


def project_card(source: dict[str, Any]) -> dict[str, Any]:
    result = pick(
        source,
        (
            "id",
            "name",
            "geometry",
            "coordinate_source",
            "elevation_m",
            "provider_start_date",
            "provider_end_date",
            "provider_status",
            "in_official_catalog",
            "has_local_data",
            "has_example",
        ),
    )
    result["schema_version"] = 1
    result["has_giano_results"] = source["has_benchmark_results"]
    result["variables"] = {
        variable: {
            "unit": value["unit"],
            "data_coverage": pick(
                value["data_coverage"],
                (
                    "observed_hours",
                    "hours_between_first_and_last",
                    "natural_missing_hours_inside_coverage",
                    "first_observation",
                    "last_observation",
                    "timezone",
                ),
            ),
            "giano_results": project_metrics(value["benchmark"]),
            "results_status": "available"
            if value["benchmark"] is not None
            else "not_evaluated_in_this_validation_run",
            "example": pick(value["example"], ("json", "image"))
            if value["example"]
            else None,
            "example_status": "available"
            if value["example"]
            else "no_eligible_validation_window",
        }
        for variable, value in source["variables"].items()
    }
    return result


def plot_example(example: dict[str, Any], output: Path) -> None:
    """Plot observations, ground truth and Giano reconstruction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    series = example["series"]
    times = np.asarray(series["timestamps"], dtype="datetime64[ns]")
    hidden = np.asarray(series["synthetic_mask"], dtype=bool)
    reconstructed = np.asarray(series["reconstruction"], dtype=float)
    # Include visible endpoints around the gap.
    segment = hidden.copy()
    segment[1:] |= hidden[:-1]
    segment[:-1] |= hidden[1:]
    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    ax.plot(times, series["ground_truth"], color="#344054", lw=1.7, label=LEGEND[0])
    ax.plot(
        times,
        series["observed_with_mask"],
        ".",
        color="#8595a7",
        markersize=4,
        label=LEGEND[1],
    )
    ax.plot(
        times,
        np.where(segment, reconstructed, np.nan),
        color="#00877c",
        lw=2,
        label=LEGEND[2],
    )
    for value in times[hidden]:
        ax.axvspan(
            value - np.timedelta64(30, "m"),
            value + np.timedelta64(30, "m"),
            color="#db8e36",
            alpha=0.12,
            lw=0,
        )
    ax.set_title(
        f"Giano · {example['station']} · {example['variable'].replace('_', ' ').title()} · synthetic gap (shaded)"
    )
    ax.set_ylabel(example["unit"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", fontsize=8)
    metrics = example["metrics"]
    fig.supxlabel(
        f"{str(times[0])[:10]} — {str(times[-1])[:10]} · validation · seed {example['seed']} · "
        f"{metrics['hidden_points']} hidden points · MAE {metrics['mae']:.3g} {example['unit']}",
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, metadata={"Description": "; ".join(LEGEND)})
    plt.close(fig)


def verify_bundle(bundle: Path) -> dict[str, int]:
    """Verify file hashes, links, series and scores."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("kind") != "giano_station_presentation":
        raise ValueError("Not a station-presentation bundle")
    actual = {
        p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()
    }
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Unlisted or missing presentation files")
    for name in actual:
        path = safe_path(bundle, name)
        if name != "manifest.json" and file_sha256(path) != manifest["files"][name]:
            raise ValueError(f"Changed presentation file: {name}")
        if path.suffix in {".json", ".geojson", ".md"} and FORBIDDEN.search(
            path.read_text()
        ):
            raise ValueError(f"Out-of-scope presentation content: {name}")
    features = json.loads((bundle / "stations.geojson").read_text())["features"]
    if len({f["id"] for f in features}) != len(features):
        raise ValueError("Duplicate station ID")
    examples = 0
    for feature in features:
        card = json.loads(safe_path(bundle, feature["properties"]["card"]).read_text())
        if card["id"] != feature["id"] or card["geometry"] != feature["geometry"]:
            raise ValueError("Map/card mismatch")
        for variable, item in card["variables"].items():
            if item["example"] is None:
                continue
            example = json.loads(safe_path(bundle, item["example"]["json"]).read_text())
            if example["station"] != card["id"] or example["variable"] != variable:
                raise ValueError("Card/example mismatch")
            safe_path(bundle, item["example"]["image"]).read_bytes()
            series = example["series"]
            arrays = {
                key: np.asarray(series[key], dtype=float)
                for key in (
                    "ground_truth",
                    "observed_with_mask",
                    "reconstruction",
                )
            }
            mask = np.asarray(series["synthetic_mask"], dtype=bool)
            unknown = np.asarray(series["ground_truth_unavailable"], dtype=bool)
            if any(
                len(series[k]) != 72 for k in SERIES_FIELDS if k != "timestamp_timezone"
            ):
                raise ValueError("Unaligned time series")
            target, visible, prediction = (
                arrays[k]
                for k in ("ground_truth", "observed_with_mask", "reconstruction")
            )
            if (
                not mask.any()
                or np.any(mask & unknown)
                or not np.isnan(target[unknown]).all()
                or not np.isnan(prediction[unknown]).all()
                or not np.isnan(visible[mask | unknown]).all()
                or not np.array_equal(
                    target[~(mask | unknown)], visible[~(mask | unknown)]
                )
                or not np.array_equal(
                    target[~(mask | unknown)], prediction[~(mask | unknown)]
                )
                or not np.isfinite(prediction[mask]).all()
            ):
                raise ValueError("Invalid observation/ground-truth preservation")
            error = prediction[mask] - target[mask]
            if variable == "wind_direction":
                error = (error + 180) % 360 - 180
            if (
                int(mask.sum()) != example["metrics"]["hidden_points"]
                or not np.isclose(
                    np.abs(error).mean(), example["metrics"]["mae"], atol=1e-5
                )
                or not np.isclose(
                    np.sqrt(np.square(error).mean()),
                    example["metrics"]["rmse"],
                    atol=1e-5,
                )
            ):
                raise ValueError("Giano scores disagree with exported values")
            examples += 1
    if (
        len(features) != manifest["counts"]["stations"]
        or examples != manifest["counts"]["examples"]
    ):
        raise ValueError("Incorrect presentation counts")
    return {"stations": len(features), "examples": examples, "files": len(actual)}


def build(source: Path, destination: Path) -> dict[str, int]:
    """Build station assets from selected fields and regenerated plots."""
    original = json.loads((source / "manifest.json").read_text())

    def read(name: str) -> dict[str, Any]:
        path = safe_path(source, name)
        if original["files"].get(name) != file_sha256(path):
            raise ValueError(f"Source hash mismatch: {name}")
        return json.loads(path.read_text())

    write_json(destination / "results.json", project_results(read("results.json")))
    features = []
    images = 0
    for feature in read("stations.geojson")["features"]:
        link = feature["properties"]["card"]
        card = project_card(read(link))
        write_json(safe_path(destination, link), card)
        properties = pick(
            feature["properties"],
            (
                "id",
                "name",
                "card",
                "variables",
                "coordinate_source",
                "provider_status",
                "in_official_catalog",
                "has_local_data",
                "has_example",
            ),
        )
        properties["has_giano_results"] = card["has_giano_results"]
        features.append(
            {
                "type": "Feature",
                "id": card["id"],
                "geometry": card["geometry"],
                "properties": properties,
            }
        )
        for item in card["variables"].values():
            if item["example"] is None:
                continue
            example = project_example(read(item["example"]["json"]))
            write_json(safe_path(destination, item["example"]["json"]), example)
            plot_example(example, safe_path(destination, item["example"]["image"]))
            images += 1
        if images and images % 50 == 0:
            print(f"Rendered {images} Giano examples", flush=True)
    write_json(
        destination / "stations.geojson",
        {"type": "FeatureCollection", "features": features},
    )
    write_json(
        destination / "content.json",
        {
            "en": {
                "title": "Giano: reconstructing meteorological observations",
                "description": "Explore Meteotrentino stations and the missing observations reconstructed by Giano.",
                "results_description": "Validation results for each weather variable, measured on artificially masked observations.",
                "example_description": "A 72-hour window showing observations and Giano's reconstruction. Errors refer to the masked points.",
            },
            "it": {
                "title": "Giano: ricostruzione delle osservazioni meteorologiche",
                "description": "Esplorare le stazioni Meteotrentino e i dati mancanti ricostruiti da Giano.",
                "results_description": "Risultati di validazione per variabile, misurati su osservazioni nascoste artificialmente.",
                "example_description": "Una finestra di 72 ore con osservazioni e ricostruzione di Giano. Gli errori si riferiscono ai punti mascherati.",
            },
        },
    )
    counts = {
        "stations": len(features),
        "examples": images,
        "stations_with_data": sum(f["properties"]["has_local_data"] for f in features),
        "stations_with_giano_results": sum(
            f["properties"]["has_giano_results"] for f in features
        ),
        "stations_with_examples": sum(f["properties"]["has_example"] for f in features),
    }
    write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "kind": "giano_station_presentation",
            "model": "Giano",
            "built_at": datetime.now(UTC).isoformat(),
            "counts": counts,
            "station_catalog": pick(
                original["station_catalog"],
                ("source_url", "fetched_at", "attribution", "sha256"),
            ),
            "source_manifest_sha256": file_sha256(source / "manifest.json"),
            "builder_sha256": file_sha256(Path(__file__)),
            "files": {
                p.relative_to(destination).as_posix(): file_sha256(p)
                for p in sorted(destination.rglob("*"))
                if p.is_file()
            },
        },
    )
    return verify_bundle(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--archive", type=Path, default=Path("artifacts/website/giano-stations.zip")
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_bundle(args.output), indent=2))
        return
    if args.output.exists() or args.archive.exists():
        raise FileExistsError(
            "Preserve existing output/archive; choose a new destination"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".giano-presentation-", dir=args.output.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        checks = build(args.source, staging)
        staging.rename(args.output)
    with ZipFile(args.archive, "x", compression=ZIP_DEFLATED) as archive:
        for path in sorted(args.output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(args.output.parent))
    with ZipFile(args.archive) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity failure")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "archive": str(args.archive.resolve()),
                **checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
