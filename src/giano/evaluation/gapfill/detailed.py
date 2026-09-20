"""Paired window-by-station analysis for model selection and uncertainty."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from giano.baselines.train_bilstm import load_bilstm_checkpoint
from giano.evaluation.gapfill.cases import (
    BenchmarkCase,
    dataset_for_case,
    load_benchmark_protocol,
)
from giano.evaluation.gapfill.results import EvaluationUnit, evaluate_model_detailed
from giano.experiments import bilstm_checkpoint_path, imputeformer_checkpoint_path
from giano.model.train_imputeformer import (
    TrainingConfig,
    _loader,
    load_imputeformer_checkpoint,
)
from giano.prediction import prediction_policy
from giano.provenance import git_provenance
from giano.runtime import default_device
from giano.spatiotemporal_dataset import build_spatiotemporal_panel
from giano.variables import VARIABLE_TYPE_NAMES


@dataclass(frozen=True)
class PairedInterval:
    """Paired aggregate-MAE difference over window-by-station units."""

    variable: str
    mask_type: str
    mask_parameter: int | float
    candidate: str
    reference: str
    units: int
    hidden_points: int
    mean_difference: float
    ci_low: float
    ci_high: float


def _index(units: list[EvaluationUnit]) -> dict[tuple[int, str], EvaluationUnit]:
    indexed = {(unit.start_index, unit.station): unit for unit in units}
    if len(indexed) != len(units):
        raise ValueError("detailed evaluation contains duplicate paired units")
    return indexed


def pair_units(
    giano_units: list[EvaluationUnit],
    bilstm_units: list[EvaluationUnit],
    *,
    variable: str,
    seed: int,
    case: BenchmarkCase,
) -> list[dict[str, Any]]:
    """Align two model evaluations and verify their interpolation reference."""
    giano = _index(giano_units)
    bilstm = _index(bilstm_units)
    if giano.keys() != bilstm.keys():
        raise ValueError("Giano and BiLSTM did not evaluate identical station windows")
    paired: list[dict[str, Any]] = []
    for key in sorted(giano):
        first = giano[key]
        second = bilstm[key]
        if first.n_hidden != second.n_hidden or not np.allclose(
            [first.baseline_abs_sum, first.baseline_squared_sum],
            [second.baseline_abs_sum, second.baseline_squared_sum],
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError("paired models have different masks or baselines")
        row: dict[str, Any] = {
            "variable": variable,
            "seed": seed,
            "mask_type": case.mask_type,
            "mask_parameter": case.parameter,
            "mask_parameter_unit": case.parameter_unit,
            "start_index": first.start_index,
            "station": first.station,
            "n_hidden": first.n_hidden,
            "giano_abs_sum": first.model_abs_sum,
            "giano_squared_sum": first.model_squared_sum,
            "bilstm_fair_abs_sum": second.model_abs_sum,
            "bilstm_fair_squared_sum": second.model_squared_sum,
            "interpolation_abs_sum": first.baseline_abs_sum,
            "interpolation_squared_sum": first.baseline_squared_sum,
        }
        if variable == "precipitation":
            row.update(
                {
                    "target_sum": first.target_sum,
                    "target_peak": first.target_peak,
                    "giano_sum": first.model_sum,
                    "giano_peak": first.model_peak,
                    "giano_wet_true_positive": first.model_wet_true_positive,
                    "giano_wet_false_positive": first.model_wet_false_positive,
                    "giano_wet_false_negative": first.model_wet_false_negative,
                    "bilstm_fair_sum": second.model_sum,
                    "bilstm_fair_peak": second.model_peak,
                    "bilstm_fair_wet_true_positive": second.model_wet_true_positive,
                    "bilstm_fair_wet_false_positive": second.model_wet_false_positive,
                    "bilstm_fair_wet_false_negative": second.model_wet_false_negative,
                    "interpolation_sum": first.baseline_sum,
                    "interpolation_peak": first.baseline_peak,
                    "interpolation_wet_true_positive": first.baseline_wet_true_positive,
                    "interpolation_wet_false_positive": first.baseline_wet_false_positive,
                    "interpolation_wet_false_negative": first.baseline_wet_false_negative,
                }
            )
        paired.append(row)
    return paired


def paired_unit_interval(
    rows: list[dict[str, Any]],
    *,
    candidate: str,
    reference: str,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Hierarchically resample paired seeds, then station-window units."""
    if not rows or samples <= 0:
        raise ValueError("paired bootstrap requires rows and positive samples")
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    seed_values = sorted(by_seed)

    arrays = {
        item_seed: (
            np.asarray(
                [row[f"{candidate}_abs_sum"] for row in group],
                dtype=np.float64,
            ),
            np.asarray(
                [row[f"{reference}_abs_sum"] for row in group],
                dtype=np.float64,
            ),
            np.asarray([row["n_hidden"] for row in group], dtype=np.int64),
        )
        for item_seed, group in by_seed.items()
    }
    total_count = sum(int(row["n_hidden"]) for row in rows)
    observed = (
        sum(float(row[f"{candidate}_abs_sum"]) for row in rows) / total_count
        - sum(float(row[f"{reference}_abs_sum"]) for row in rows) / total_count
    )
    rng = np.random.default_rng(seed)
    bootstrapped = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        selected_seeds = rng.choice(seed_values, len(seed_values), replace=True)
        candidate_sum = reference_sum = 0.0
        sampled_count = 0
        for selected_seed in selected_seeds:
            candidate_values, reference_values, counts = arrays[int(selected_seed)]
            indices = rng.integers(0, len(counts), size=len(counts))
            candidate_sum += float(candidate_values[indices].sum())
            reference_sum += float(reference_values[indices].sum())
            sampled_count += int(counts[indices].sum())
        bootstrapped[sample] = (
            candidate_sum / sampled_count - reference_sum / sampled_count
        )
    low, high = np.quantile(bootstrapped, (0.025, 0.975))
    return observed, float(low), float(high)


def _station_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["variable"],
                row["seed"],
                row["mask_type"],
                row["mask_parameter"],
                row["station"],
            )
        ].append(row)
    output = []
    for (variable, seed, mask_type, parameter, station), group in sorted(
        grouped.items()
    ):
        count = sum(int(row["n_hidden"]) for row in group)
        item: dict[str, Any] = {
            "variable": variable,
            "seed": seed,
            "mask_type": mask_type,
            "mask_parameter": parameter,
            "station": station,
            "n_hidden": count,
        }
        for model in ("giano", "bilstm_fair", "interpolation"):
            absolute = sum(float(row[f"{model}_abs_sum"]) for row in group)
            squared = sum(float(row[f"{model}_squared_sum"]) for row in group)
            item[f"{model}_mae"] = absolute / count
            item[f"{model}_rmse"] = math.sqrt(squared / count)
        output.append(item)
    return output


def _precipitation_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    precipitation = [row for row in rows if row["variable"] == "precipitation"]
    grouped: dict[tuple[str, int | float], list[dict[str, Any]]] = defaultdict(list)
    for row in precipitation:
        grouped[(str(row["mask_type"]), row["mask_parameter"])].append(row)
    output = []
    for (mask_type, parameter), group in sorted(grouped.items()):
        target_sum = sum(float(row["target_sum"]) for row in group)
        for model in ("giano", "bilstm_fair", "interpolation"):
            true_positive = sum(int(row[f"{model}_wet_true_positive"]) for row in group)
            false_positive = sum(
                int(row[f"{model}_wet_false_positive"]) for row in group
            )
            false_negative = sum(
                int(row[f"{model}_wet_false_negative"]) for row in group
            )
            precision = true_positive / max(true_positive + false_positive, 1)
            recall = true_positive / max(true_positive + false_negative, 1)
            output.append(
                {
                    "mask_type": mask_type,
                    "mask_parameter": parameter,
                    "model": model,
                    "wet_threshold": 0.1,
                    "wet_precision": precision,
                    "wet_recall": recall,
                    "wet_f1": 2 * precision * recall / max(precision + recall, 1e-12),
                    "volume_bias": (
                        sum(float(row[f"{model}_sum"]) for row in group) - target_sum
                    )
                    / max(target_sum, 1e-12),
                    "mean_peak_bias": float(
                        np.mean(
                            [
                                float(row[f"{model}_peak"]) - float(row["target_peak"])
                                for row in group
                            ]
                        )
                    ),
                }
            )
    return output


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Detailed paired gap-filling analysis",
        "",
        "Intervals resample training/mask seeds and paired station-window units. "
        "Negative candidate-minus-Giano MAE differences favor the candidate.",
        "",
        "| Variable | Mask | Candidate | Delta MAE | 95% CI | Units | Hidden |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    for item in payload["paired_intervals"]:
        lines.append(
            f"| {item['variable']} | {item['mask_type']} {item['mask_parameter']} | "
            f"{item['candidate']} | {item['mean_difference']:.4f} | "
            f"[{item['ci_low']:.4f}, {item['ci_high']:.4f}] | "
            f"{item['units']} | {item['hidden_points']} |"
        )
    if payload["precipitation_event_metrics"]:
        lines.extend(
            [
                "",
                "## Precipitation event diagnostics",
                "",
                "| Mask | Model | Wet F1 | Volume bias | Mean peak bias |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for item in payload["precipitation_event_metrics"]:
            lines.append(
                f"| {item['mask_type']} {item['mask_parameter']} | {item['model']} | "
                f"{item['wet_f1']:.3f} | {item['volume_bias']:.3f} | "
                f"{item['mean_peak_bias']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the detailed validation analysis."""
    parser = argparse.ArgumentParser(
        description="Evaluate paired window-by-station errors and uncertainty"
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/2-processed-v2"))
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=VARIABLE_TYPE_NAMES,
        default=VARIABLE_TYPE_NAMES,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--imputeformer-checkpoint-dir",
        type=Path,
        default=Path("checkpoints/imputeformer"),
    )
    parser.add_argument(
        "--bilstm-checkpoint-dir",
        type=Path,
        default=Path("checkpoints/bilstm"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/evaluation/gapfill/analysis/validation_detailed.json.gz"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0914
    """Run paired Giano/BiLSTM/interpolation validation and persist its units."""
    args = parse_args(argv)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("detailed-analysis seeds must be non-empty and unique")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    protocol = load_benchmark_protocol(args.config)
    max_batches = args.max_batches or protocol.max_batches
    device: torch.device = default_device()
    units: list[dict[str, Any]] = []
    checkpoints: dict[str, str] = {}
    for variable in args.variables:
        panel = build_spatiotemporal_panel(args.data_dir, variable)
        for seed in args.seeds:
            giano_path = imputeformer_checkpoint_path(
                args.imputeformer_checkpoint_dir,
                variable,
                seed=seed,
            )
            bilstm_path = bilstm_checkpoint_path(
                args.bilstm_checkpoint_dir,
                "fair",
                variable,
                seed=seed,
            )
            giano, giano_metadata = load_imputeformer_checkpoint(giano_path, device)
            bilstm, bilstm_metadata = load_bilstm_checkpoint(bilstm_path, device)
            if bilstm_metadata.get("variable") != variable:
                raise ValueError("BiLSTM checkpoint variable mismatch")
            config = replace(TrainingConfig.from_checkpoint(giano_metadata), seed=seed)
            checkpoints[f"giano/seed-{seed}/{variable}"] = str(giano_path)
            checkpoints[f"bilstm/fair/seed-{seed}/{variable}"] = str(bilstm_path)
            for case in protocol.cases:
                dataset = dataset_for_case(panel, args.split, config, case, seed)
                _giano_result, giano_units = evaluate_model_detailed(
                    giano,
                    _loader(dataset, config, shuffle=False),
                    device,
                    variable,
                    panel.station_ids,
                    max_batches=max_batches,
                )
                _bilstm_result, bilstm_units = evaluate_model_detailed(
                    bilstm,
                    _loader(dataset, config, shuffle=False),
                    device,
                    variable,
                    panel.station_ids,
                    max_batches=max_batches,
                )
                units.extend(
                    pair_units(
                        giano_units,
                        bilstm_units,
                        variable=variable,
                        seed=seed,
                        case=case,
                    )
                )
    grouped_cases: dict[tuple[str, str, int | float], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in units:
        grouped_cases[
            (row["variable"], row["mask_type"], row["mask_parameter"])
        ].append(row)
    intervals: list[PairedInterval] = []
    for case_index, ((variable, mask_type, parameter), group) in enumerate(
        sorted(grouped_cases.items())
    ):
        for candidate in ("bilstm_fair", "interpolation"):
            observed, low, high = paired_unit_interval(
                group,
                candidate=candidate,
                reference="giano",
                samples=args.bootstrap_samples,
                seed=17 + case_index,
            )
            intervals.append(
                PairedInterval(
                    variable=variable,
                    mask_type=mask_type,
                    mask_parameter=parameter,
                    candidate=candidate,
                    reference="giano",
                    units=len(group),
                    hidden_points=sum(int(row["n_hidden"]) for row in group),
                    mean_difference=observed,
                    ci_low=low,
                    ci_high=high,
                )
            )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": {
            "prediction_postprocessing": prediction_policy(),
            "split": args.split,
            "variables": list(args.variables),
            "seeds": list(args.seeds),
            "cases": [asdict(case) for case in protocol.cases],
            "max_batches": max_batches,
            "bootstrap": "hierarchical paired seed then station-window unit",
            "bootstrap_samples": args.bootstrap_samples,
            "checkpoints": checkpoints,
            "code": git_provenance(Path.cwd().resolve()),
        },
        "paired_intervals": [asdict(item) for item in intervals],
        "station_metrics": _station_metrics(units),
        "precipitation_event_metrics": _precipitation_metrics(units),
        "units": units,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)
    markdown = args.output.with_suffix("").with_suffix(".md")
    markdown.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "markdown": str(markdown),
                "units": len(units),
                "station_metrics": len(payload["station_metrics"]),
                "paired_intervals": len(intervals),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
