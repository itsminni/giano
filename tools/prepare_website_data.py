"""Build station data, research results and reconstruction plots for the website."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from giano.model.train_imputeformer import TrainingConfig, load_imputeformer_checkpoint
from giano.prediction import physical_prediction, prediction_policy
from giano.provenance import file_sha256, training_dataset_identity
from giano.spatiotemporal_dataset import (
    SpatiotemporalPanel,
    SpatiotemporalWindowDataset,
    build_spatiotemporal_panel,
)
from giano.variables import VARIABLE_TYPE_NAMES

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notebooks.release_artifacts import (  # noqa: E402
    DOWNSTREAM_PATH,
    RELEASE_PATH,
    SEEDS,
    load_benchmarks,
    read_completed,
)

CATALOG_URL = "https://dati.meteotrentino.it/service.asmx/listaStazioniGeoJson"
UNITS = {
    "temperature": "degC",
    "humidity": "%",
    "pressure": "hPa",
    "precipitation": "mm",
    "wind_speed": "m/s",
    "wind_direction": "degrees",
}
METHODS = {
    "giano/imputeformer": {"label": "Giano", "original_2025_model": False},
    "bilstm/fair": {"label": "BiLSTM fair (retrained)", "original_2025_model": False},
    "bilstm/legacy_retrained": {
        "label": "Legacy-style BiLSTM (retrained)",
        "original_2025_model": False,
    },
    "interpolation": {"label": "Temporal interpolation", "original_2025_model": False},
}


def write_json(path: Path, payload: Any) -> None:
    """Write strict UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def improvement(model_mae: float, reference_mae: float) -> float | None:
    """Return MAE reduction in percent, or None for a zero reference."""
    return (
        100.0 * (reference_mae - model_mae) / reference_mae
        if reference_mae > 0
        else None
    )


def normalize_catalog(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize all catalog stations, including historical and unmapped ones."""
    if payload.get("type") != "FeatureCollection":
        raise ValueError("Expected official station GeoJSON FeatureCollection")
    stations = {}
    for feature in payload["features"]:
        raw = feature["properties"]
        code = raw["codice"]
        if not isinstance(code, str) or not code or not code.isalnum():
            raise ValueError("Unsafe or missing official station code")
        if code in stations:
            raise ValueError(f"Duplicate official station code: {code}")
        geometry = feature.get("geometry")
        if geometry is not None:
            coordinates = geometry.get("coordinates", [])
            if (
                geometry.get("type") != "Point"
                or len(coordinates) != 2
                or not np.isfinite(coordinates).all()
                or not -180 <= coordinates[0] <= 180
                or not -90 <= coordinates[1] <= 90
            ):
                raise ValueError(f"Invalid official station geometry: {code}")
        elevation = raw.get("quota")
        stations[code] = {
            "id": code,
            "name": raw.get("nome") or code,
            "geometry": geometry,
            "coordinate_source": "Meteotrentino official station catalog"
            if geometry
            else None,
            "elevation_m": float(elevation) if elevation not in (None, "") else None,
            "provider_start_date": raw.get("inizio") or None,
            "provider_end_date": raw.get("fine") or None,
            "provider_status": "historical"
            if raw.get("fine")
            else "no_end_date_reported",
            "in_official_catalog": True,
            "variables": {},
        }
    return stations


def benchmark_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Average paired benchmark scores across seeds and cases."""
    variables = {}
    for variable in VARIABLE_TYPE_NAMES:
        selected = [row for row in rows if row["variable"] == variable]
        scores = {}
        for method in METHODS:
            subset = [
                row
                for row in selected
                if (
                    "interpolation"
                    if row["model"] == "interpolation"
                    else f"{row['model']}/{row['model_variant']}"
                )
                == method
            ]
            scores[method] = {
                "mae": float(np.mean([r["mae"] for r in subset])),
                "mean_case_seed_rmse": float(np.mean([r["rmse"] for r in subset])),
                "case_seed_groups": len(subset),
            }
        giano = scores["giano/imputeformer"]["mae"]
        variables[variable] = {
            "unit": UNITS[variable],
            "scores": scores,
            "giano_mae_reduction_percent": {
                method: improvement(giano, score["mae"])
                for method, score in scores.items()
                if method != "giano/imputeformer"
            },
            "original_2025_supported": variable in {"temperature", "wind_speed"},
            "original_2025_paired_comparison": None,
        }
    return {
        "schema_version": 1,
        "split": "val",
        "seeds": list(SEEDS),
        "aggregation": "equal weights across 13 cases and five paired seeds",
        "improvement_formula": "100 * (reference_mae - giano_mae) / reference_mae; positive is better",
        "methods": METHODS,
        "variables": variables,
        "original_2025": {
            "supported_variables": ["temperature", "wind_speed"],
            "paired_comparison_status": "not_available_in_this_corrected_protocol",
            "note": "Retrained baselines; original 2025 weights excluded.",
        },
        "limitations": [
            "Validation also used during development.",
            "Errors measured on synthetic gaps, separately by variable.",
            "Counts include overlapping windows and repeated masks.",
        ],
    }


def station_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the station's evaluated case/seed groups."""
    keys = [(row["seed"], row["mask_type"], row["mask_parameter"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate station metric groups")
    scores = {
        method: {
            "mae": float(np.mean([r[f"{method}_mae"] for r in rows])),
            "mean_case_seed_rmse": float(np.mean([r[f"{method}_rmse"] for r in rows])),
        }
        for method in ("giano", "bilstm_fair", "interpolation")
    }
    return {
        "split": "val",
        "aggregation": "equal weight to available case/seed groups",
        "available_case_seed_groups": len(rows),
        "expected_case_seed_groups": 65,
        "complete_case_seed_coverage": len(rows) == 65,
        "hidden_evaluations_including_repeats": sum(r["n_hidden"] for r in rows),
        "scores": scores,
        "giano_mae_reduction_percent": {
            method: improvement(scores["giano"]["mae"], scores[method]["mae"])
            for method in ("bilstm_fair", "interpolation")
        },
        "case_seed_scores": rows,
    }


class StationExampleDataset(SpatiotemporalWindowDataset):
    """Include the selected station in illustrative validation windows."""

    target_node: int = 0

    def _select_nodes(
        self,
        observed: np.ndarray,
        rng: np.random.Generator,
        natural_missing: np.ndarray | None = None,
        required_node_indices: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        nodes = super()._select_nodes(
            observed, rng, natural_missing, required_node_indices
        )
        if self.target_node in nodes:
            return nodes
        return np.sort(
            np.concatenate(([self.target_node], nodes[: self.max_nodes - 1]))
        )


def nullable(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


@torch.no_grad()
def example_payload(
    model: torch.nn.Module,
    dataset: StationExampleDataset,
    node: int,
) -> dict[str, Any] | None:
    """Sample an eligible validation window independently of model errors."""
    dataset.target_node = node
    panel = dataset.panel
    station = panel.station_ids[node]
    digest = hashlib.sha256(f"{panel.variable}/{station}/42".encode()).digest()
    random_seed = int.from_bytes(digest[:4], "little")
    finite = np.isfinite(panel.values[:, node])
    counts = np.concatenate(([0], np.cumsum(finite)))
    eligible = np.flatnonzero(
        counts[dataset.starts + dataset.seq_len] - counts[dataset.starts] >= 24
    )
    for index in np.random.default_rng(random_seed).permutation(eligible):
        sample = dataset[int(index)]
        local = int(torch.nonzero(sample["node_indices"] == node)[0, 0])
        hidden = sample["evaluation_mask"][:, local].numpy().astype(bool)
        visible = sample["visible_mask"][:, local].numpy().astype(bool)
        if not hidden.any() or visible.sum() < dataset.min_context_points:
            continue
        prediction, _ = model(
            *[
                sample[key].unsqueeze(0)
                for key in ("features", "coordinates", "node_mask", "baseline")
            ]
        )
        reconstructed = physical_prediction(
            prediction,
            sample["center"][None],
            sample["scale"][None],
            variable=panel.variable,
        )[0, :, local].numpy()
        target = sample["target_physical"][:, local].numpy()
        baseline = sample["baseline_physical"][:, local].numpy()
        observed = visible | hidden
        error = reconstructed[hidden] - target[hidden]
        baseline_error = baseline[hidden] - target[hidden]
        if panel.variable == "wind_direction":
            error = (error + 180) % 360 - 180
            baseline_error = (baseline_error + 180) % 360 - 180
        return {
            "schema_version": 1,
            "station": station,
            "variable": panel.variable,
            "unit": UNITS[panel.variable],
            "split": "val",
            "seed": 42,
            "selection_seed": random_seed,
            "sample_index": int(index),
            "start_index": int(sample["start_index"]),
            "selection": "seeded random eligible validation window; target station included",
            "illustration_only": True,
            "included_in_headline_scores": False,
            "case": {"mask_type": "block", "requested_length_hours": 12},
            "model_context_stations": [
                panel.station_ids[int(n)] for n in sample["node_indices"] if n >= 0
            ],
            "model_coordinate_source": "auxiliary_grid_proxy",
            "prediction_postprocessing": prediction_policy(),
            "metrics": {
                "scope": "this synthetic mask only",
                "hidden_points": int(hidden.sum()),
                "giano_mae": float(np.abs(error).mean()),
                "giano_rmse": float(np.sqrt(np.square(error).mean())),
                "interpolation_mae": float(np.abs(baseline_error).mean()),
            },
            "series": {
                "timestamps": sample["timestamps"]
                .numpy()
                .astype("datetime64[ns]")
                .astype(str)
                .tolist(),
                "timestamp_timezone": "not established by source",
                "ground_truth": nullable(np.where(observed, target, np.nan)),
                "observed_with_mask": nullable(np.where(visible, target, np.nan)),
                "reconstruction": nullable(
                    np.where(visible, target, np.where(hidden, reconstructed, np.nan))
                ),
                "interpolation": nullable(
                    np.where(visible, target, np.where(hidden, baseline, np.nan))
                ),
                "synthetic_mask": hidden.tolist(),
                "ground_truth_unavailable": (~observed).tolist(),
            },
        }
    return None


def plot_example(example: dict[str, Any], output: Path) -> None:
    """Plot the reconstruction, mask and ground truth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    series = example["series"]
    times = np.asarray(series["timestamps"], dtype="datetime64[ns]")
    hidden = np.asarray(series["synthetic_mask"])
    fig, ax = plt.subplots(figsize=(10, 3.7), layout="constrained")
    ax.plot(
        times, series["ground_truth"], color="#243247", lw=1.7, label="Ground truth"
    )
    ax.plot(
        times,
        series["observed_with_mask"],
        ".",
        color="#8894a4",
        markersize=4,
        label="Visible observations",
    )
    ax.plot(
        times,
        series["interpolation"],
        "--",
        color="#b57426",
        lw=1,
        label="Interpolation",
    )
    ax.plot(
        times,
        series["reconstruction"],
        color="#00877c",
        lw=1.6,
        label="Giano reconstruction",
    )
    for time_value in times[hidden]:
        ax.axvspan(
            time_value - np.timedelta64(30, "m"),
            time_value + np.timedelta64(30, "m"),
            color="#db8e36",
            alpha=0.12,
            lw=0,
        )
    ax.set_title(
        f"{example['station']} · {example['variable'].replace('_', ' ').title()} · synthetic gap (shaded)"
    )
    ax.set_ylabel(example["unit"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    metrics = example["metrics"]
    fig.supxlabel(
        f"Illustrative validation window · seed 42 · MAE on {metrics['hidden_points']} hidden points: {metrics['giano_mae']:.3g} {example['unit']}",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=140,
        metadata={"Software": "Giano reproducible website data exporter"},
    )
    plt.close(fig)


def data_coverage(panel: SpatiotemporalPanel, node: int) -> dict[str, Any]:
    finite = np.flatnonzero(np.isfinite(panel.values[:, node]))
    if not len(finite):
        return {
            "observed_hours": 0,
            "first_observation": None,
            "last_observation": None,
        }
    first, last = int(finite[0]), int(finite[-1])
    return {
        "observed_hours": len(finite),
        "hours_between_first_and_last": last - first + 1,
        "natural_missing_hours_inside_coverage": last - first + 1 - len(finite),
        "first_observation": str(panel.times[first]),
        "last_observation": str(panel.times[last]),
        "timezone": "not established",
        "coordinate_source_used_by_model": "auxiliary_grid_proxy",
    }


def add_variable(
    root: Path,
    destination: Path,
    stations: dict[str, dict[str, Any]],
    variable: str,
    identity: dict[str, Any],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Process detailed results one variable at a time."""
    detailed_path = root / RELEASE_PATH / "detailed" / variable / "validation.json.gz"
    detailed, detailed_identity = read_completed(detailed_path)
    if detailed_identity != identity or detailed["protocol"]["split"] != "val":
        raise ValueError("Detailed scores do not match the common validation snapshot")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detailed["station_metrics"]:
        if row["variable"] != variable:
            raise ValueError("Wrong variable in detailed station scores")
        groups[row["station"]].append(row)
    extra = {
        key: detailed[key]
        for key in ("paired_intervals", "precipitation_event_metrics")
    }
    del detailed
    sources.append(
        {
            "path": str(detailed_path.relative_to(root)),
            "sha256": file_sha256(detailed_path),
        }
    )
    checkpoint = (
        root
        / "checkpoints/corrected_v2_spectral/imputeformer/seed-42"
        / f"{variable}.pt"
    )
    if (
        training_dataset_identity(root / "data/2-processed-v2", variable)
        != identity["datasets"][variable]
    ):
        raise ValueError("Example data differ from the scored release")
    if (
        file_sha256(checkpoint)
        != identity["checkpoints"][str(checkpoint.relative_to(root))]
    ):
        raise ValueError("Example checkpoint differs from the scored release")
    model, metadata = load_imputeformer_checkpoint(checkpoint, torch.device("cpu"))
    config = TrainingConfig.from_checkpoint(metadata)
    panel = build_spatiotemporal_panel(root / "data/2-processed-v2", variable)
    dataset = StationExampleDataset(
        panel,
        "val",
        seq_len=config.seq_len,
        stride=config.seq_len,
        max_nodes=config.max_nodes,
        min_context_points=config.min_context_points,
        seed=42,
        mask_mode="block",
        block_lengths=(12,),
    )
    for node, code in enumerate(panel.station_ids):
        if not code.isalnum():
            raise ValueError("Unsafe dataset station code")
        if code not in stations:
            stations[code] = {
                "id": code,
                "name": code,
                "geometry": None,
                "coordinate_source": None,
                "in_official_catalog": False,
                "provider_status": "unknown",
                "variables": {},
            }
        example = example_payload(model, dataset, node)
        example_link = None
        if example is not None:
            example["checkpoint_sha256"] = file_sha256(checkpoint)
            example_link = {
                "json": f"examples/{code}/{variable}.json",
                "image": f"images/{code}/{variable}.png",
            }
            write_json(destination / example_link["json"], example)
            plot_example(example, destination / example_link["image"])
        stations[code]["variables"][variable] = {
            "unit": UNITS[variable],
            "data_coverage": data_coverage(panel, node),
            "benchmark": station_scores(groups[code]) if code in groups else None,
            "benchmark_status": "evaluated"
            if code in groups
            else "not_selected_by_frozen_validation_node_sampler",
            "example": example_link,
            "example_status": "available"
            if example_link
            else "no_eligible_validation_window_with_24_observations_and_hidden_target",
        }
    if set(groups) - set(panel.station_ids):
        raise ValueError("Detailed metrics reference absent dataset stations")
    print(
        f"Website assets: {variable}, {panel.num_nodes} station series, {len(groups)} benchmark stations",
        flush=True,
    )
    return extra


def build(root: Path, destination: Path, catalog_file: Path | None) -> dict[str, Any]:
    """Build the website data bundle."""
    fetched_at = None
    if catalog_file is None:
        with urllib.request.urlopen(CATALOG_URL, timeout=45) as response:  # noqa: S310 - fixed official HTTPS URL
            catalog_bytes = response.read()
        fetched_at = datetime.now(UTC).isoformat()
    else:
        catalog_bytes = catalog_file.read_bytes()
    stations = normalize_catalog(json.loads(catalog_bytes))
    (destination / "sources").mkdir(parents=True)
    (destination / "sources/meteotrentino_stations.geojson").write_bytes(catalog_bytes)
    rows, paths = load_benchmarks(root)
    identity = json.loads(Path(paths[0] + ".complete.json").read_text())["request"][
        "inputs"
    ]
    sources = [
        {"path": str(Path(p).relative_to(root)), "sha256": file_sha256(Path(p))}
        for p in paths
    ]
    summary = benchmark_summary(rows)
    summary["detailed"] = {}
    for variable in VARIABLE_TYPE_NAMES:
        summary["detailed"][variable] = add_variable(
            root, destination, stations, variable, identity, sources
        )
    write_json(destination / "results.json", summary)
    features = []
    for code, station in sorted(stations.items()):
        station["schema_version"] = 1
        station["has_local_data"] = bool(station["variables"])
        station["has_benchmark_results"] = any(
            v["benchmark"] is not None for v in station["variables"].values()
        )
        station["has_example"] = any(
            v["example"] is not None for v in station["variables"].values()
        )
        write_json(destination / "stations" / f"{code}.json", station)
        features.append(
            {
                "type": "Feature",
                "id": code,
                "geometry": station["geometry"],
                "properties": {
                    "id": code,
                    "name": station["name"],
                    "card": f"stations/{code}.json",
                    "variables": list(station["variables"]),
                    **{
                        key: station[key]
                        for key in (
                            "coordinate_source",
                            "provider_status",
                            "in_official_catalog",
                            "has_local_data",
                            "has_benchmark_results",
                            "has_example",
                        )
                    },
                },
            }
        )
    write_json(
        destination / "stations.geojson",
        {"type": "FeatureCollection", "features": features},
    )
    # Downstream results use a separate provenance snapshot.
    downstream = {}
    for variable in VARIABLE_TYPE_NAMES:
        path = root / DOWNSTREAM_PATH / "downstream" / f"{variable}.json"
        if not path.with_name(path.name + ".complete.json").exists():
            downstream[variable] = {"status": "pending"}
            continue
        payload, downstream_identity = read_completed(
            path, postprocessing_key="repair_prediction_postprocessing"
        )
        if any(
            downstream_identity[key] != identity[key]
            for key in ("datasets", "checkpoints", "config_sha256", "lock_sha256")
        ):
            raise ValueError(
                "Downstream data/model recipe differs from the scored release"
            )
        downstream[variable] = {
            "status": payload["protocol"]["evaluation_status"],
            "stations": payload["protocol"]["stations"],
            "unavailable_variables": payload["protocol"]["unavailable_variables"],
            "summaries": payload["summaries"],
            "paired_method_comparisons": payload.get("paired_method_comparisons", []),
        }
        sources.append(
            {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
        )
    write_json(
        destination / "downstream.json",
        {
            "schema_version": 1,
            "split": "test",
            "not_for_model_selection": True,
            "scope": "one availability-selected station per evaluable variable",
            "variables": downstream,
        },
    )
    manifest = {
        "schema_version": 1,
        "built_at": datetime.now(UTC).isoformat(),
        "website_implemented": False,
        "policy_status": "deferred",
        "code_license_status": "owner_choice_pending",
        "station_catalog": {
            "source_url": CATALOG_URL,
            "fetched_at": fetched_at,
            "attribution": "Provincia autonoma di Trento — Meteotrentino",
            "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
            "scope": "official catalog plus local stations; missing coordinates are null",
            "publication_note": "Data redistribution review pending.",
        },
        "counts": {
            "official_stations": sum(
                s["in_official_catalog"] for s in stations.values()
            ),
            "cards": len(stations),
            "mapped_stations": sum(
                s["geometry"] is not None for s in stations.values()
            ),
            "local_data_stations": sum(s["has_local_data"] for s in stations.values()),
            "benchmark_stations": sum(
                s["has_benchmark_results"] for s in stations.values()
            ),
            "example_stations": sum(s["has_example"] for s in stations.values()),
            "example_station_variable_pairs": sum(
                v["example"] is not None
                for s in stations.values()
                for v in s["variables"].values()
            ),
        },
        "sources": sources,
        "evaluation_input_identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
        "builder_sha256": file_sha256(Path(__file__)),
        "files": {
            str(p.relative_to(destination)): file_sha256(p)
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        },
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def verify_bundle(bundle: Path) -> dict[str, int]:
    """Verify checksums, links, masks and example scores."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    for relative, digest in manifest["files"].items():
        path = bundle / relative
        if (
            not path.resolve().is_relative_to(bundle.resolve())
            or file_sha256(path) != digest
        ):
            raise ValueError(f"Website asset checksum/path mismatch: {relative}")
    features = json.loads((bundle / "stations.geojson").read_text())["features"]
    if len({f["id"] for f in features}) != len(features):
        raise ValueError("Duplicate map identifiers")
    examples = 0
    for feature in features:
        card = json.loads((bundle / feature["properties"]["card"]).read_text())
        if card["id"] != feature["id"] or card["geometry"] != feature["geometry"]:
            raise ValueError("Map/card join mismatch")
        for variable, entry in card["variables"].items():
            if entry["example"] is None:
                continue
            payload = json.loads((bundle / entry["example"]["json"]).read_text())
            if payload["station"] != card["id"] or payload["variable"] != variable:
                raise ValueError("Example/card join mismatch")
            if not (bundle / entry["example"]["image"]).is_file():
                raise ValueError("Missing example plot")
            series = payload["series"]
            target = np.asarray(series["ground_truth"], dtype=float)
            visible = np.asarray(series["observed_with_mask"], dtype=float)
            predicted = np.asarray(series["reconstruction"], dtype=float)
            hidden = np.asarray(series["synthetic_mask"], dtype=bool)
            unknown = np.asarray(series["ground_truth_unavailable"], dtype=bool)
            if len(target) != 72 or any(
                len(series[key]) != 72
                for key in (
                    "timestamps",
                    "reconstruction",
                    "interpolation",
                    "synthetic_mask",
                    "ground_truth_unavailable",
                    "observed_with_mask",
                )
            ):
                raise ValueError("Example series are not aligned")
            if (
                np.any(hidden & unknown)
                or not np.isnan(target[unknown]).all()
                or not np.isnan(predicted[unknown]).all()
                or not np.isnan(visible[hidden | unknown]).all()
                or not np.array_equal(
                    target[~(hidden | unknown)], predicted[~(hidden | unknown)]
                )
                or not np.isfinite(predicted[hidden]).all()
                or not hidden.any()
            ):
                raise ValueError(
                    "Example changed observations or invented missing truth"
                )
            error = predicted[hidden] - target[hidden]
            if variable == "wind_direction":
                error = (error + 180) % 360 - 180
            if not np.isclose(
                np.abs(error).mean(), payload["metrics"]["giano_mae"], atol=1e-5
            ):
                raise ValueError("Example score does not match exported series")
            examples += 1
    if examples != manifest["counts"]["example_station_variable_pairs"]:
        raise ValueError("Example count differs from manifest")
    return {
        "cards_checked": len(features),
        "examples_checked": examples,
        "checksums_checked": len(manifest["files"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/website/corrected_v2_release")
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        help="Saved station GeoJSON (offline mode)",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    torch.set_num_threads(1)
    output = args.output.resolve()
    if args.verify_only:
        print(json.dumps(verify_bundle(output), indent=2))
        return
    if output.exists():
        raise FileExistsError(f"Preserve existing website bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".website-build-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        manifest = build(root, staging, args.catalog)
        verify_bundle(staging)
        staging.rename(output)
    print(json.dumps({"output": str(output), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
