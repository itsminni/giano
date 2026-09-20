"""Run the frozen downstream forecasting utility experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from giano.downstream.ridge_forecaster import (
    AutoregressiveRidgeForecaster,
    complete_origins,
)
from giano.evaluation.forecasting.checkpoint_repairer import checkpoint_history_repairer
from giano.evaluation.forecasting.protocol import (
    ForecastRecord,
    ForecastSummary,
    evaluate_rolling_origin,
    load_forecasting_config,
    paired_method_comparisons,
    summarize_forecasts,
    write_forecasting_artifact,
)
from giano.evaluation.gapfill.cases import BenchmarkCase
from giano.evaluation.gapfill.masks import empirical_block_lengths
from giano.experiments import bilstm_checkpoint_path, imputeformer_checkpoint_path
from giano.prediction import prediction_policy
from giano.provenance import git_provenance
from giano.runtime import default_device
from giano.spatiotemporal_dataset import SpatiotemporalPanel, build_spatiotemporal_panel
from giano.variables import VARIABLE_TYPE_NAMES

DEFAULT_CASES = ("point:0.3", "block:12", "terminal:12", "empirical:0.95")


class NoEligibleStationError(ValueError):
    """No station satisfies the frozen downstream data-availability protocol."""


def parse_case(value: str) -> BenchmarkCase:
    """Parse ``mask:value`` without allowing implicit post-result selection."""
    try:
        mask_type, raw_parameter = value.split(":", maxsplit=1)
        parameter = float(raw_parameter)
    except ValueError as error:
        raise argparse.ArgumentTypeError("cases must use mask:value") from error
    if mask_type not in {
        "point",
        "block",
        "spatial_block",
        "terminal",
        "empirical",
    }:
        raise argparse.ArgumentTypeError(f"unsupported downstream mask: {mask_type}")
    numeric: int | float = (
        parameter if mask_type in {"point", "empirical"} else int(parameter)
    )
    if mask_type not in {"point", "empirical"} and parameter != numeric:
        raise argparse.ArgumentTypeError("block durations must be whole hours")
    try:
        return BenchmarkCase(mask_type, numeric)  # type: ignore[arg-type]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _complete_test_origins(
    values: np.ndarray,
    *,
    test_start: int,
    history_length: int,
    horizons: tuple[int, ...],
    stride: int,
) -> int:
    return len(
        complete_origins(
            values,
            history_length=history_length,
            horizons=horizons,
            start=test_start,
            end=len(values),
            stride=stride,
        )
    )


def select_station(
    panel: SpatiotemporalPanel,
    *,
    history_length: int,
    horizons: tuple[int, ...],
    stride: int,
    requested: str | None,
) -> tuple[int, int]:
    """Maximize complete test origins among stations with >=2 training origins.

    Only data availability is used; ties retain panel order, never model scores.
    """
    test_start = int(len(panel.times) * 0.85)
    training_counts = [
        len(
            complete_origins(
                panel.values[:, node],
                history_length=history_length,
                horizons=horizons,
                start=0,
                end=int(len(panel.times) * 0.7),
            )
        )
        for node in range(panel.num_nodes)
    ]
    if requested is not None:
        try:
            index = panel.station_ids.index(requested)
        except ValueError as error:
            raise ValueError(f"station {requested} is unavailable") from error
        if training_counts[index] < 2:
            raise ValueError(
                f"station {requested} has fewer than two complete training origins"
            )
        count = _complete_test_origins(
            panel.values[:, index],
            test_start=test_start,
            history_length=history_length,
            horizons=horizons,
            stride=stride,
        )
        if count == 0:
            raise ValueError(f"station {requested} has no complete test origins")
        return index, count
    counts = [
        _complete_test_origins(
            panel.values[:, node],
            test_start=test_start,
            history_length=history_length,
            horizons=horizons,
            stride=stride,
        )
        if training_counts[node] >= 2
        else 0
        for node in range(panel.num_nodes)
    ]
    index = int(np.argmax(counts))
    if counts[index] == 0:
        raise NoEligibleStationError(
            f"no {panel.variable} station has both complete training and test origins"
        )
    return index, counts[index]


def _markdown(summaries: list[ForecastSummary]) -> str:
    lines = [
        "# Downstream forecasting utility",
        "",
        "Paired differences are method MAE minus corrupted-history MAE on the "
        "same rolling origins. Negative values are better.",
        "",
        "| Variable | Station | Seed | Mask | Method | Horizon | MAE | RMSE | "
        "Recovery | Delta vs corrupted | 95% paired CI |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        recovery = (
            "n/a" if item.recovery_ratio is None else f"{item.recovery_ratio:.3f}"
        )
        difference = (
            "n/a"
            if item.mae_difference_vs_corrupted is None
            else f"{item.mae_difference_vs_corrupted:.4f}"
        )
        interval = (
            "n/a"
            if item.difference_ci_low is None or item.difference_ci_high is None
            else f"[{item.difference_ci_low:.4f}, {item.difference_ci_high:.4f}]"
        )
        parameter = (
            f"{float(item.mask_parameter):.0%}"
            if item.mask_parameter_unit in {"fraction", "quantile"}
            else f"{int(item.mask_parameter)} h"
        )
        lines.append(
            f"| {item.variable} | {item.station} | {item.seed} | "
            f"{item.mask_type} {parameter} | {item.method} | {item.horizon} | "
            f"{item.mae:.4f} | {item.rmse:.4f} | {recovery} | {difference} | "
            f"{interval} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the downstream experiment request."""
    parser = argparse.ArgumentParser(
        description="Compare checkpoint repairs through a frozen Ridge forecaster"
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/2-processed-v2"))
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=VARIABLE_TYPE_NAMES,
        default=("temperature",),
    )
    parser.add_argument("--station")
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--cases", nargs="+", type=parse_case, default=DEFAULT_CASES)
    parser.add_argument("--max-origins", type=int)
    parser.add_argument(
        "--record-unavailable",
        action="store_true",
        help="Record absent eligible stations explicitly, without scores",
    )
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
        default=Path("artifacts/evaluation/forecasting/downstream_utility.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0914
    """Evaluate clean/corrupted/interpolated and checkpoint-repaired histories."""
    args = parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("downstream seeds must be non-empty and unique")
    cases = tuple(
        parse_case(item) if isinstance(item, str) else item for item in args.cases
    )
    ridge_config, rolling_config = load_forecasting_config(args.config)
    if args.max_origins is not None:
        rolling_config = replace(rolling_config, max_origins=args.max_origins)
        rolling_config.validate()
    device: torch.device = default_device()
    records: list[ForecastRecord] = []
    checkpoints: dict[str, str] = {}
    stations: dict[str, dict[str, int | str]] = {}
    unavailable: dict[str, dict[str, object]] = {}
    for variable in args.variables:
        panel = build_spatiotemporal_panel(args.data_dir, variable)
        try:
            station_index, available_origins = select_station(
                panel,
                history_length=ridge_config.history_length,
                horizons=ridge_config.horizons,
                stride=rolling_config.stride,
                requested=args.station,
            )
        except NoEligibleStationError as error:
            if not args.record_unavailable:
                raise
            unavailable[variable] = {
                "reason": str(error),
                "station_availability": [
                    {
                        "station": name,
                        "complete_training_origins": len(
                            complete_origins(
                                panel.values[:, node],
                                history_length=ridge_config.history_length,
                                horizons=ridge_config.horizons,
                                start=0,
                                end=int(len(panel.times) * 0.7),
                            )
                        ),
                        "complete_test_origins": _complete_test_origins(
                            panel.values[:, node],
                            test_start=int(len(panel.times) * 0.85),
                            history_length=ridge_config.history_length,
                            horizons=ridge_config.horizons,
                            stride=rolling_config.stride,
                        ),
                    }
                    for node, name in enumerate(panel.station_ids)
                ],
            }
            continue
        station = panel.station_ids[station_index]
        stations[variable] = {
            "station": station,
            "available_complete_origins": available_origins,
            "available_complete_training_origins": len(
                complete_origins(
                    panel.values[:, station_index],
                    history_length=ridge_config.history_length,
                    horizons=ridge_config.horizons,
                    start=0,
                    end=int(len(panel.times) * 0.7),
                )
            ),
        }
        train_end = int(len(panel.times) * 0.7)
        test_start = int(len(panel.times) * 0.85)
        empirical_by_case = {
            case: empirical_block_lengths(
                panel.values,
                panel.coverage_bounds,
                train_end,
                quantile_cap=float(case.parameter),
                # The mask builder requires two visible history points. Its
                # upper bound is exclusive, so 72-hour histories cap at 70.
                max_length=ridge_config.history_length - 1,
            )
            for case in cases
            if case.mask_type == "empirical"
        }
        for seed in args.seeds:
            imputeformer_path = imputeformer_checkpoint_path(
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
            if not imputeformer_path.is_file() or not bilstm_path.is_file():
                raise FileNotFoundError(
                    f"missing paired downstream checkpoints for {variable}/seed-{seed}"
                )
            checkpoints[f"giano/seed-{seed}/{variable}"] = str(imputeformer_path)
            checkpoints[f"bilstm/fair/seed-{seed}/{variable}"] = str(bilstm_path)
            repairers = {
                "giano": checkpoint_history_repairer(
                    imputeformer_path,
                    panel,
                    station,
                    family="imputeformer",
                    device=device,
                ),
                "bilstm_fair": checkpoint_history_repairer(
                    bilstm_path,
                    panel,
                    station,
                    family="bilstm",
                    device=device,
                ),
            }
            for case in cases:
                records.extend(
                    evaluate_rolling_origin(
                        panel.values[:, station_index],
                        panel.times,
                        train_end=train_end,
                        test_start=test_start,
                        forecaster=AutoregressiveRidgeForecaster(ridge_config),
                        case=case,
                        config=replace(rolling_config, seed=seed),
                        direction=variable == "wind_direction",
                        repairers=repairers,
                        empirical_lengths=empirical_by_case.get(case),
                        variable=variable,
                        station=station,
                        experiment_seed=seed,
                    )
                )
    summaries = summarize_forecasts(
        records,
        bootstrap_samples=rolling_config.bootstrap_samples,
        seed=rolling_config.seed,
    )
    comparisons = paired_method_comparisons(
        records,
        reference="giano",
        candidates=("bilstm_fair", "interpolation"),
        bootstrap_samples=rolling_config.bootstrap_samples,
        seed=rolling_config.seed,
    )
    protocol: dict[str, object] = {
        "repair_prediction_postprocessing": prediction_policy(),
        "config": str(args.config),
        "data_dir": str(args.data_dir),
        "variables": list(args.variables),
        "stations": stations,
        "evaluation_status": ("partial" if records else "not_evaluable")
        if unavailable
        else "complete",
        "unavailable_variables": unavailable,
        "station_selection": "at-least-two-clean-training-origins-then-most-complete-test-origins-v1",
        "seeds": list(args.seeds),
        "cases": [asdict(case) for case in cases],
        "ridge": asdict(ridge_config),
        "rolling_origin": asdict(rolling_config),
        "origin_selection": "first-up-to-max-complete-origins-v1",
        "minimum_visible_history_points": 2,
        "checkpoints": checkpoints,
        "future_values_visible_to_imputer": False,
        "auxiliary_values_after_origin_visible_to_imputer": False,
        "code": git_provenance(Path.cwd().resolve()),
    }
    payload = write_forecasting_artifact(
        args.output,
        records,
        summaries,
        protocol=protocol,
        comparisons=comparisons,
    )
    markdown = args.output.with_suffix(".md")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    text = _markdown(summaries)
    for variable, detail in unavailable.items():
        text += f"\nNot evaluable — {variable}: {detail['reason']}. No scores were produced.\n"
    markdown.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "markdown": str(markdown),
                "records": len(records),
                "summaries": len(summaries),
                "paired_method_comparisons": len(comparisons),
                "schema_version": payload["schema_version"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
