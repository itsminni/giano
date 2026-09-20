"""Aggregate paired gap-filling runs and describe train-only natural gaps."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from giano.evaluation.gapfill.masks import natural_gap_lengths
from giano.provenance import git_provenance
from giano.spatiotemporal_dataset import (
    SpatiotemporalPanel,
    build_spatiotemporal_panel,
)
from giano.variables import VARIABLE_TYPE_NAMES

MODEL_ORDER = (
    "giano/imputeformer",
    "bilstm/fair",
    "bilstm/legacy_retrained",
    "interpolation/linear",
    "interpolation/circular_linear",
)


@dataclass(frozen=True)
class AggregateScore:
    """Mean and sample spread across independently trained paired seeds."""

    variable: str
    model: str
    mask_type: str
    mask_parameter: int | float
    seeds: tuple[int, ...]
    mean_mae: float
    std_mae: float
    mean_rmse: float
    std_rmse: float
    mean_runtime_ms: float
    parameter_count: int


@dataclass(frozen=True)
class PairedSeedDifference:
    """Paired MAE difference using the same training and mask seed."""

    variable: str
    reference_model: str
    candidate_model: str
    mask_type: str
    mask_parameter: int | float
    seeds: tuple[int, ...]
    mean_mae_difference: float
    ci_low: float
    ci_high: float
    candidate_wins: int


def _model_label(row: dict[str, Any]) -> str:
    return f"{row['model']}/{row['model_variant']}"


def _sample_std(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def aggregate_scores(rows: Iterable[dict[str, Any]]) -> list[AggregateScore]:
    """Aggregate one row per model, seed, and case without mixing variables."""
    grouped: dict[tuple[str, str, str, int | float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["variable"]),
            _model_label(row),
            str(row["mask_type"]),
            row["mask_parameter"],
        )
        grouped.setdefault(key, []).append(row)
    aggregates: list[AggregateScore] = []
    for (variable, model, mask_type, parameter), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: int(item["seed"]))
        seeds = tuple(int(item["seed"]) for item in ordered)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seeds for {variable}/{model}/{mask_type}")
        mae = np.asarray([item["mae"] for item in ordered], dtype=np.float64)
        rmse = np.asarray([item["rmse"] for item in ordered], dtype=np.float64)
        runtime = np.asarray([item["runtime_ms"] for item in ordered], dtype=np.float64)
        aggregates.append(
            AggregateScore(
                variable=variable,
                model=model,
                mask_type=mask_type,
                mask_parameter=parameter,
                seeds=seeds,
                mean_mae=float(mae.mean()),
                std_mae=_sample_std(mae),
                mean_rmse=float(rmse.mean()),
                std_rmse=_sample_std(rmse),
                mean_runtime_ms=float(runtime.mean()),
                parameter_count=int(ordered[0]["parameter_count"]),
            )
        )
    return aggregates


def paired_seed_differences(
    rows: Iterable[dict[str, Any]],
    *,
    reference_model: str,
    candidate_model: str,
    bootstrap_samples: int,
    seed: int,
) -> list[PairedSeedDifference]:
    """Bootstrap paired seed-level MAE differences, candidate minus reference."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    indexed: dict[tuple[str, str, int | float, int, str], float] = {}
    for row in rows:
        key = (
            str(row["variable"]),
            str(row["mask_type"]),
            row["mask_parameter"],
            int(row["seed"]),
            _model_label(row),
        )
        indexed[key] = float(row["mae"])
    cases = sorted({(key[0], key[1], key[2]) for key in indexed})
    output: list[PairedSeedDifference] = []
    for case_index, (variable, mask_type, parameter) in enumerate(cases):
        reference_seeds = {
            key[3]
            for key in indexed
            if key[:3] == (variable, mask_type, parameter) and key[4] == reference_model
        }
        candidate_seeds = {
            key[3]
            for key in indexed
            if key[:3] == (variable, mask_type, parameter) and key[4] == candidate_model
        }
        paired = tuple(sorted(reference_seeds & candidate_seeds))
        if not paired:
            continue
        differences = np.asarray(
            [
                indexed[(variable, mask_type, parameter, item, candidate_model)]
                - indexed[(variable, mask_type, parameter, item, reference_model)]
                for item in paired
            ],
            dtype=np.float64,
        )
        rng = np.random.default_rng(seed + case_index)
        indices = rng.integers(
            0,
            len(differences),
            size=(bootstrap_samples, len(differences)),
        )
        bootstrap_means = differences[indices].mean(axis=1)
        low, high = np.quantile(bootstrap_means, (0.025, 0.975))
        output.append(
            PairedSeedDifference(
                variable=variable,
                reference_model=reference_model,
                candidate_model=candidate_model,
                mask_type=mask_type,
                mask_parameter=parameter,
                seeds=paired,
                mean_mae_difference=float(differences.mean()),
                ci_low=float(low),
                ci_high=float(high),
                candidate_wins=int((differences < 0).sum()),
            )
        )
    return output


def natural_gap_statistics(
    panel: SpatiotemporalPanel,
    *,
    train_ratio: float = 0.7,
) -> dict[str, Any]:
    """Describe natural missing runs inside station coverage using train only."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be inside (0, 1)")
    train_end = int(len(panel.times) * train_ratio)
    all_lengths: list[int] = []
    by_station: list[dict[str, Any]] = []
    for node, station in enumerate(panel.station_ids):
        lengths = natural_gap_lengths(
            panel.values[:, node : node + 1],
            panel.coverage_bounds[node : node + 1],
            train_end,
        ).tolist()
        all_lengths.extend(lengths)
        by_station.append(
            {
                "station": station,
                "gap_count": len(lengths),
                "missing_hours": int(sum(lengths)),
                "max_gap_hours": max(lengths, default=0),
            }
        )
    values = np.asarray(all_lengths, dtype=np.float64)
    buckets = {
        "1h": 0,
        "2h": 0,
        "3h": 0,
        "4-6h": 0,
        "7-12h": 0,
        "13-24h": 0,
        "25-48h": 0,
        ">48h": 0,
    }
    for length in all_lengths:
        if length <= 3:
            buckets[f"{length}h"] += 1
        elif length <= 6:
            buckets["4-6h"] += 1
        elif length <= 12:
            buckets["7-12h"] += 1
        elif length <= 24:
            buckets["13-24h"] += 1
        elif length <= 48:
            buckets["25-48h"] += 1
        else:
            buckets[">48h"] += 1
    quantiles = (
        {
            name: float(value)
            for name, value in zip(
                ("p50", "p75", "p90", "p95", "p99"),
                np.quantile(values, (0.5, 0.75, 0.9, 0.95, 0.99)),
                strict=True,
            )
        }
        if values.size
        else {name: None for name in ("p50", "p75", "p90", "p95", "p99")}
    )
    return {
        "variable": panel.variable,
        "split": "train",
        "train_end": np.datetime_as_string(panel.times[train_end - 1], unit="h"),
        "station_count": panel.num_nodes,
        "gap_count": len(all_lengths),
        "missing_hours": int(sum(all_lengths)),
        "max_gap_hours": max(all_lengths, default=0),
        "quantiles_hours": quantiles,
        "histogram": buckets,
        "stations": by_station,
    }


def _load_paired_rows(
    artifact_root: Path,
    variables: Iterable[str],
    seeds: tuple[int, ...],
    split: str,
) -> tuple[list[dict[str, Any]], list[str], dict[str, float | None]]:
    selected_variables = tuple(dict.fromkeys(variables))
    seed_scope = "-".join(str(item) for item in sorted(seeds))
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    postprocessing_contracts: set[str] = set()
    variation: dict[str, list[float]] = defaultdict(list)
    run_scope = f"{split}_paired-seeds-{seed_scope}"
    combined_root = artifact_root / "_".join(sorted(selected_variables)) / run_scope
    combined_paths = tuple(
        combined_root / f"{family}_benchmark.json"
        for family in ("imputeformer", "bilstm")
    )
    runs: tuple[tuple[Path, frozenset[str]], ...]
    if len(selected_variables) > 1 and all(path.is_file() for path in combined_paths):
        runs = ((combined_root, frozenset(selected_variables)),)
    else:
        runs = tuple(
            (artifact_root / variable / run_scope, frozenset((variable,)))
            for variable in selected_variables
        )
    for run_root, run_variables in runs:
        for family in ("imputeformer", "bilstm"):
            path = run_root / f"{family}_benchmark.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing paired artifact: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            protocol = payload.get("protocol", {})
            postprocessing_contracts.add(
                json.dumps(protocol.get("prediction_postprocessing"), sort_keys=True)
            )
            if len(postprocessing_contracts) > 1:
                raise ValueError(
                    "Cannot combine artifacts with different prediction postprocessing "
                    f"(including legacy raw outputs): {path}"
                )
            if (
                payload.get("schema_version") != 2
                or protocol.get("seed_mode") != "paired_training_and_mask"
                or tuple(protocol.get("seeds", ())) != seeds
            ):
                raise ValueError(f"Artifact does not match paired protocol: {path}")
            sources.append(str(path))
            for row in payload["results"]:
                if row["variable"] not in run_variables:
                    continue
                if family == "bilstm" and row["model"] == "interpolation":
                    continue
                rows.append(row)
            for comparison in payload.get("comparisons", []):
                variable = str(comparison["variable"])
                if variable not in run_variables:
                    continue
                ratio = comparison.get("variation_ratio")
                if ratio is None:
                    continue
                model = (
                    "giano/imputeformer"
                    if family == "imputeformer"
                    else f"bilstm/{comparison['model_variant']}"
                )
                variation[f"{variable}:{model}"].append(float(ratio))
    mean_variation = {
        key: float(np.mean(values)) if values else None
        for key, values in sorted(variation.items())
    }
    return rows, sources, mean_variation


def _markdown(payload: dict[str, Any]) -> str:
    aggregates = [AggregateScore(**item) for item in payload["aggregates"]]
    wins: dict[str, Counter[str]] = defaultdict(Counter)
    by_case: dict[tuple[str, str, int | float], list[AggregateScore]] = defaultdict(
        list
    )
    for score in aggregates:
        if score.model == "bilstm/legacy_retrained":
            continue
        by_case[(score.variable, score.mask_type, score.mask_parameter)].append(score)
    for (variable, _mask, _parameter), scores in by_case.items():
        winner = min(scores, key=lambda item: item.mean_mae)
        wins[variable][winner.model] += 1
    lines = [
        "# Paired multi-seed gap-filling analysis",
        "",
        "Negative paired MAE difference means the candidate beats Giano. "
        "Intervals resample only the available training/mask seeds and are not "
        "a substitute for station- or window-level bootstrap intervals.",
        "",
        "## Winners by benchmark case",
        "",
        "| Variable | Giano | BiLSTM fair | Interpolation |",
        "|---|---:|---:|---:|",
    ]
    for variable in sorted(wins):
        counter = wins[variable]
        interpolation = (
            counter["interpolation/linear"] + counter["interpolation/circular_linear"]
        )
        lines.append(
            f"| {variable} | {counter['giano/imputeformer']} | "
            f"{counter['bilstm/fair']} | {interpolation} |"
        )
    lines.extend(
        [
            "",
            "## Mean MAE across seeds",
            "",
            "| Variable | Mask | Model | MAE mean | MAE SD | RMSE mean |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    order = {model: index for index, model in enumerate(MODEL_ORDER)}
    for score in sorted(
        aggregates,
        key=lambda item: (
            item.variable,
            item.mask_type,
            float(item.mask_parameter),
            order.get(item.model, len(order)),
        ),
    ):
        if score.mask_type == "point":
            parameter = f"{float(score.mask_parameter):.0%}"
        elif score.mask_type == "empirical":
            parameter = f"train q{float(score.mask_parameter):.0%}"
        else:
            parameter = f"{int(score.mask_parameter)} h"
        lines.append(
            f"| {score.variable} | {score.mask_type} {parameter} | {score.model} | "
            f"{score.mean_mae:.4f} | {score.std_mae:.4f} | "
            f"{score.mean_rmse:.4f} |"
        )
    lines.extend(["", "## Train-only natural gaps", ""])
    lines.extend(
        [
            "| Variable | Stations | Gaps | Missing hours | P50 h | P95 h | Max h |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["natural_gaps"]:
        quantiles = item["quantiles_hours"]
        lines.append(
            f"| {item['variable']} | {item['station_count']} | {item['gap_count']} | "
            f"{item['missing_hours']} | {quantiles['p50']} | {quantiles['p95']} | "
            f"{item['max_gap_hours']} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse analysis arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate paired gap-filling runs and train-only natural gaps"
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/evaluation/gapfill")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/2-processed-v2"))
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=VARIABLE_TYPE_NAMES,
        default=VARIABLE_TYPE_NAMES,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write machine-readable and Markdown summaries from existing artifacts."""
    args = parse_args(argv)
    seeds = tuple(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("analysis seeds must be non-empty and unique")
    rows, sources, variation = _load_paired_rows(
        args.artifact_root, args.variables, seeds, args.split
    )
    aggregates = aggregate_scores(rows)
    differences = []
    for candidate in (
        "bilstm/fair",
        "interpolation/linear",
        "interpolation/circular_linear",
    ):
        differences.extend(
            paired_seed_differences(
                rows,
                reference_model="giano/imputeformer",
                candidate_model=candidate,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
        )
    natural_gaps = [
        natural_gap_statistics(build_spatiotemporal_panel(args.data_dir, variable))
        for variable in args.variables
    ]
    project_root = Path.cwd().resolve()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": {
            "split": args.split,
            "seeds": list(seeds),
            "bootstrap_unit": "paired_training_and_mask_seed",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "station_performance_available": False,
            "source_artifacts": sources,
            "code": git_provenance(project_root),
        },
        "aggregates": [asdict(item) for item in aggregates],
        "paired_seed_differences": [asdict(item) for item in differences],
        "mean_variation_ratio": variation,
        "natural_gaps": natural_gaps,
    }
    seed_scope = "-".join(str(item) for item in sorted(seeds))
    root = args.artifact_root / "analysis" / f"{args.split}_paired-seeds-{seed_scope}"
    output = args.output or root / "summary.json"
    markdown_output = args.markdown_output or output.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"output": str(output), "markdown": str(markdown_output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
