"""Common result schema and artifact writers for gap-filling benchmarks."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from giano.evaluation.gapfill.cases import BenchmarkCase
from giano.prediction import physical_prediction, prediction_policy

BENCHMARK_ARTIFACT_ROOT = Path("artifacts/evaluation/gapfill")


def benchmark_artifact_paths(
    model_family: str,
    variables: Iterable[str],
    *,
    split: str,
    seeds: Iterable[int],
    seed_mode: str = "mask_only",
    output: Path | None = None,
    markdown_output: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve collision-free default paths for one benchmark run."""
    variable_names = sorted(set(variables))
    seed_values = sorted(set(seeds))
    if not variable_names:
        raise ValueError("benchmark paths require at least one variable")
    if not seed_values:
        raise ValueError("benchmark paths require at least one seed")
    if not model_family or any(character in model_family for character in "/\\"):
        raise ValueError("model_family must be a path-safe name")
    if seed_mode not in {"mask_only", "paired_training_and_mask"}:
        raise ValueError("unsupported benchmark seed mode")
    variable_scope = "_".join(variable_names)
    seed_scope = "-".join(str(seed) for seed in seed_values)
    run_scope = (
        f"{split}_paired-seeds-{seed_scope}"
        if seed_mode == "paired_training_and_mask"
        else f"{split}_seeds-{seed_scope}"
    )
    default_dir = BENCHMARK_ARTIFACT_ROOT / variable_scope / run_scope
    json_path = output or default_dir / f"{model_family}_benchmark.json"
    if markdown_output is not None:
        markdown_path = markdown_output
    elif output is not None:
        markdown_path = output.with_suffix(".md")
    else:
        markdown_path = default_dir / f"{model_family}_benchmark.md"
    return json_path, markdown_path


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated same-mask model and interpolation scores."""

    model_mae: float
    baseline_mae: float
    model_rmse: float
    baseline_rmse: float
    variation_ratio: float
    count: int

    @property
    def beats_baseline(self) -> bool:
        """Return whether model MAE is strictly smaller."""
        return self.model_mae < self.baseline_mae


@dataclass(frozen=True)
class EvaluationUnit:
    """Error sums for one exactly paired window and station."""

    start_index: int
    station: str
    n_hidden: int
    model_abs_sum: float
    baseline_abs_sum: float
    model_squared_sum: float
    baseline_squared_sum: float
    target_sum: float | None = None
    model_sum: float | None = None
    baseline_sum: float | None = None
    target_peak: float | None = None
    model_peak: float | None = None
    baseline_peak: float | None = None
    model_wet_true_positive: int | None = None
    model_wet_false_positive: int | None = None
    model_wet_false_negative: int | None = None
    baseline_wet_true_positive: int | None = None
    baseline_wet_false_positive: int | None = None
    baseline_wet_false_negative: int | None = None


def normalized_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    direction: bool,
) -> torch.Tensor:
    """Return scalar or circular error in normalized model units."""
    error = prediction - target
    if direction:
        error = torch.remainder(error + 1.0, 2.0) - 1.0
    return error


def physical_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    direction: bool,
) -> torch.Tensor:
    """Return scalar or shortest-arc error in physical units."""
    error = prediction - target
    if direction:
        error = torch.remainder(error + 180.0, 360.0) - 180.0
    return error


def to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move every tensor in a benchmark batch to one device."""
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def _evaluate_model(
    model: Any,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    variable: str,
    *,
    max_batches: int,
    station_ids: tuple[str, ...] | None,
) -> tuple[EvaluationResult, list[EvaluationUnit]]:
    """Evaluate a common-interface model and optionally retain paired units."""
    model.eval()
    direction = variable == "wind_direction"
    model_abs = baseline_abs = model_squared = baseline_squared = 0.0
    target_variation = prediction_variation = 0.0
    count = 0
    units: list[EvaluationUnit] = []
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = to_device(cpu_batch, device)
        prediction_norm, _ = model(
            batch["features"],
            batch["coordinates"],
            batch["node_mask"],
            batch["baseline"],
        )
        prediction = physical_prediction(
            prediction_norm,
            batch["center"],
            batch["scale"],
            variable=variable,
        )
        baseline = batch["baseline_physical"]
        target = batch["target_physical"]
        mask = batch["evaluation_mask"].bool()
        model_error = physical_error(prediction, target, direction=direction)[mask]
        baseline_error = physical_error(baseline, target, direction=direction)[mask]
        model_abs += float(model_error.abs().sum().cpu())
        baseline_abs += float(baseline_error.abs().sum().cpu())
        model_squared += float(model_error.square().sum().cpu())
        baseline_squared += float(baseline_error.square().sum().cpu())
        count += int(mask.sum().cpu())
        if station_ids is not None:
            for batch_row in range(mask.shape[0]):
                start_index = int(batch["start_index"][batch_row].cpu())
                for local_node in range(mask.shape[2]):
                    active = mask[batch_row, :, local_node]
                    unit_count = int(active.sum().cpu())
                    if unit_count == 0:
                        continue
                    global_node = int(
                        batch["node_indices"][batch_row, local_node].cpu()
                    )
                    if not 0 <= global_node < len(station_ids):
                        raise RuntimeError("detailed evaluation found invalid station")
                    model_unit = physical_error(
                        prediction[batch_row, :, local_node],
                        target[batch_row, :, local_node],
                        direction=direction,
                    )[active]
                    baseline_unit = physical_error(
                        baseline[batch_row, :, local_node],
                        target[batch_row, :, local_node],
                        direction=direction,
                    )[active]
                    target_unit = target[batch_row, :, local_node][active]
                    prediction_unit = prediction[batch_row, :, local_node][active]
                    baseline_values = baseline[batch_row, :, local_node][active]
                    precipitation = variable == "precipitation"
                    target_wet = target_unit >= 0.1 if precipitation else None
                    model_wet = prediction_unit >= 0.1 if precipitation else None
                    baseline_wet = baseline_values >= 0.1 if precipitation else None
                    units.append(
                        EvaluationUnit(
                            start_index=start_index,
                            station=station_ids[global_node],
                            n_hidden=unit_count,
                            model_abs_sum=float(model_unit.abs().sum().cpu()),
                            baseline_abs_sum=float(baseline_unit.abs().sum().cpu()),
                            model_squared_sum=float(model_unit.square().sum().cpu()),
                            baseline_squared_sum=float(
                                baseline_unit.square().sum().cpu()
                            ),
                            target_sum=(
                                float(target_unit.sum().cpu())
                                if precipitation
                                else None
                            ),
                            model_sum=(
                                float(prediction_unit.sum().cpu())
                                if precipitation
                                else None
                            ),
                            baseline_sum=(
                                float(baseline_values.sum().cpu())
                                if precipitation
                                else None
                            ),
                            target_peak=(
                                float(target_unit.max().cpu())
                                if precipitation
                                else None
                            ),
                            model_peak=(
                                float(prediction_unit.max().cpu())
                                if precipitation
                                else None
                            ),
                            baseline_peak=(
                                float(baseline_values.max().cpu())
                                if precipitation
                                else None
                            ),
                            model_wet_true_positive=(
                                int((target_wet & model_wet).sum().cpu())
                                if target_wet is not None and model_wet is not None
                                else None
                            ),
                            model_wet_false_positive=(
                                int((~target_wet & model_wet).sum().cpu())
                                if target_wet is not None and model_wet is not None
                                else None
                            ),
                            model_wet_false_negative=(
                                int((target_wet & ~model_wet).sum().cpu())
                                if target_wet is not None and model_wet is not None
                                else None
                            ),
                            baseline_wet_true_positive=(
                                int((target_wet & baseline_wet).sum().cpu())
                                if target_wet is not None and baseline_wet is not None
                                else None
                            ),
                            baseline_wet_false_positive=(
                                int((~target_wet & baseline_wet).sum().cpu())
                                if target_wet is not None and baseline_wet is not None
                                else None
                            ),
                            baseline_wet_false_negative=(
                                int((target_wet & ~baseline_wet).sum().cpu())
                                if target_wet is not None and baseline_wet is not None
                                else None
                            ),
                        )
                    )
        if not direction:
            for batch_row in range(mask.shape[0]):
                active = mask[batch_row]
                if int(active.sum()) < 2:
                    continue
                target_std = target[batch_row][active].std(unbiased=False)
                pred_std = prediction[batch_row][active].std(unbiased=False)
                if target_std > 1e-8:
                    target_variation += float(target_std.cpu())
                    prediction_variation += float(pred_std.cpu())
    if count == 0:
        raise RuntimeError("Evaluation produced no scored values")
    variation_ratio = (
        prediction_variation / target_variation if target_variation > 0 else math.nan
    )
    return (
        EvaluationResult(
            model_mae=model_abs / count,
            baseline_mae=baseline_abs / count,
            model_rmse=math.sqrt(model_squared / count),
            baseline_rmse=math.sqrt(baseline_squared / count),
            variation_ratio=variation_ratio,
            count=count,
        ),
        units,
    )


def evaluate_model(
    model: Any,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    variable: str,
    *,
    max_batches: int,
) -> EvaluationResult:
    """Evaluate any common-interface gap filler on identical hidden values."""
    result, _units = _evaluate_model(
        model,
        loader,
        device,
        variable,
        max_batches=max_batches,
        station_ids=None,
    )
    return result


def evaluate_model_detailed(
    model: Any,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    variable: str,
    station_ids: tuple[str, ...],
    *,
    max_batches: int,
) -> tuple[EvaluationResult, list[EvaluationUnit]]:
    """Evaluate and retain exactly paired window-by-station error sums."""
    return _evaluate_model(
        model,
        loader,
        device,
        variable,
        max_batches=max_batches,
        station_ids=station_ids,
    )


@dataclass(frozen=True)
class BenchmarkRow:
    """One model score on one materialized family of hidden values."""

    model: str
    model_variant: str
    variable: str
    seed: int
    split: str
    station_group: str
    mask_type: str
    mask_parameter: int | float
    mask_parameter_unit: str
    mae: float
    rmse: float
    n_hidden: int
    runtime_ms: float
    parameter_count: int

    def validate(self) -> None:
        """Validate fields before publishing a machine-readable artifact."""
        if not self.model or not self.model_variant or not self.variable:
            raise ValueError("benchmark model and variable names cannot be empty")
        if self.split not in {"val", "test"}:
            raise ValueError("benchmark split must be val or test")
        if self.n_hidden <= 0:
            raise ValueError("benchmark rows must score at least one hidden point")
        if min(self.mae, self.rmse, self.runtime_ms) < 0:
            raise ValueError("benchmark metrics and runtime cannot be negative")
        if self.parameter_count < 0:
            raise ValueError("parameter_count cannot be negative")


def rows_from_comparison(
    comparison: dict[str, Any],
    *,
    model: str,
    model_variant: str,
    variable: str,
    seed: int,
    split: str,
    case: BenchmarkCase,
    runtime_ms: float,
    parameter_count: int,
) -> tuple[BenchmarkRow, BenchmarkRow]:
    """Convert a same-mask model/interpolation comparison to common rows."""
    model_row = BenchmarkRow(
        model=model,
        model_variant=model_variant,
        variable=variable,
        seed=seed,
        split=split,
        station_group="all",
        mask_type=case.mask_type,
        mask_parameter=case.parameter,
        mask_parameter_unit=case.parameter_unit,
        mae=float(comparison["model_mae"]),
        rmse=float(comparison["model_rmse"]),
        n_hidden=int(comparison["count"]),
        runtime_ms=runtime_ms,
        parameter_count=parameter_count,
    )
    interpolation_row = BenchmarkRow(
        model="interpolation",
        model_variant=("circular_linear" if variable == "wind_direction" else "linear"),
        variable=variable,
        seed=seed,
        split=split,
        station_group="all",
        mask_type=case.mask_type,
        mask_parameter=case.parameter,
        mask_parameter_unit=case.parameter_unit,
        mae=float(comparison["baseline_mae"]),
        rmse=float(comparison["baseline_rmse"]),
        n_hidden=int(comparison["count"]),
        runtime_ms=0.0,
        parameter_count=0,
    )
    model_row.validate()
    interpolation_row.validate()
    return model_row, interpolation_row


def comparison_payload(
    result: EvaluationResult,
    *,
    variable: str,
    split: str,
    seed: int,
    case: BenchmarkCase,
    runtime_ms: float,
) -> dict[str, Any]:
    """Describe model and interpolation metrics for one benchmark case."""
    payload: dict[str, Any] = asdict(result)
    if not math.isfinite(result.variation_ratio):
        payload["variation_ratio"] = None
    payload.update(
        {
            "variable": variable,
            "seed": seed,
            "split": split,
            "mask_mode": case.mask_type,
            "point_rate": (
                float(case.parameter) if case.mask_type == "point" else None
            ),
            "block_length": (
                int(case.parameter) if case.mask_type != "point" else None
            ),
            "runtime_ms": runtime_ms,
            "mae_improvement_pct": (
                (result.baseline_mae - result.model_mae) / result.baseline_mae * 100.0
                if result.baseline_mae > 0
                else None
            ),
        }
    )
    return payload


def _json_optional_metrics(value: Any) -> Any:
    """Represent undefined variation ratios as null, including training history.

    Other non-finite values remain errors for the strict JSON encoder so an
    invalid MAE or RMSE cannot be silently published as a missing metric.
    """
    if isinstance(value, dict):
        return {
            key: (
                None
                if key == "variation_ratio"
                and isinstance(item, float)
                and not math.isfinite(item)
                else _json_optional_metrics(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_optional_metrics(item) for item in value]
    return value


def write_benchmark_artifact(
    path: Path,
    rows: Iterable[BenchmarkRow],
    *,
    protocol: dict[str, Any],
    comparisons: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write deterministic JSON with common rows and per-case comparisons."""
    materialized = list(rows)
    for row in materialized:
        row.validate()
    payload: dict[str, Any] = {
        "schema_version": 2,
        "protocol": {**protocol, "prediction_postprocessing": prediction_policy()},
        "results": [asdict(row) for row in materialized],
    }
    if comparisons is not None:
        payload["comparisons"] = comparisons
    payload = _json_optional_metrics(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def markdown_table(rows: Iterable[BenchmarkRow]) -> str:
    """Generate a compact Markdown table from machine-readable result rows."""
    materialized = list(rows)
    header = (
        "| Model | Variable | Seed | Mask | MAE | RMSE | Hidden | Runtime ms |\n"
        "|---|---|---:|---|---:|---:|---:|---:|"
    )
    body = []
    for row in materialized:
        parameter = (
            f"{float(row.mask_parameter):.0%}"
            if row.mask_parameter_unit in {"fraction", "quantile"}
            else f"{int(row.mask_parameter)} h"
        )
        body.append(
            "| "
            f"{row.model}/{row.model_variant} | {row.variable} | {row.seed} | "
            f"{row.mask_type} {parameter} | {row.mae:.4f} | {row.rmse:.4f} | "
            f"{row.n_hidden} | {row.runtime_ms:.2f} |"
        )
    return "\n".join((header, *body)) + "\n"
