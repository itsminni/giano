"""Benchmark ImputeFormer under the shared, frozen missingness protocol."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch

from giano.evaluation.gapfill.cases import (
    BenchmarkCase,
    dataset_for_case,
    load_benchmark_protocol,
    protocol_with_overrides,
)
from giano.evaluation.gapfill.masks import SyntheticMaskType
from giano.evaluation.gapfill.results import (
    BenchmarkRow,
    benchmark_artifact_paths,
    comparison_payload,
    evaluate_model,
    markdown_table,
    rows_from_comparison,
    write_benchmark_artifact,
)
from giano.experiments import imputeformer_checkpoint_path
from giano.model.train_imputeformer import (
    TrainingConfig,
    _loader,
    load_imputeformer_checkpoint,
)
from giano.provenance import git_provenance
from giano.runtime import default_device
from giano.spatiotemporal_dataset import (
    MaskMode,
    SpatiotemporalPanel,
    build_spatiotemporal_panel,
)
from giano.variables import VARIABLE_TYPE_NAMES

logger = logging.getLogger(__name__)


def _benchmark_loaded_model(
    model: torch.nn.Module,
    metadata: dict[str, Any],
    panel: SpatiotemporalPanel,
    split: str,
    *,
    case: BenchmarkCase,
    seed: int,
    max_batches: int,
    device: torch.device,
) -> tuple[dict[str, Any], tuple[BenchmarkRow, BenchmarkRow]]:
    variable = metadata.get("variable")
    if not isinstance(variable, str) or variable not in VARIABLE_TYPE_NAMES:
        raise ValueError("Checkpoint contains an invalid variable")
    config = replace(TrainingConfig.from_checkpoint(metadata), seed=seed)
    dataset = dataset_for_case(panel, split, config, case, seed)
    started = time.perf_counter()
    result = evaluate_model(
        model,  # type: ignore[arg-type]
        _loader(dataset, config, shuffle=False),
        device,
        variable,
        max_batches=max_batches,
    )
    runtime_ms = (time.perf_counter() - started) * 1000.0
    comparison = comparison_payload(
        result,
        variable=variable,
        split=split,
        seed=seed,
        case=case,
        runtime_ms=runtime_ms,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    rows = rows_from_comparison(
        comparison,
        model="giano",
        model_variant="imputeformer",
        variable=variable,
        seed=seed,
        split=split,
        case=case,
        runtime_ms=runtime_ms,
        parameter_count=parameter_count,
    )
    return comparison, rows


def benchmark_checkpoint(  # noqa: PLR0913
    checkpoint: Path,
    data_root: Path,
    split: str,
    *,
    mode: MaskMode,
    point_rate: float = 0.3,
    block_length: int = 6,
    max_batches: int = 200,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint on a synthetic missingness case."""
    if mode not in {"point", "block", "spatial_block", "terminal", "empirical"}:
        raise ValueError(f"Benchmark mode must be synthetic, got {mode}")
    target_device = device or default_device()
    model, metadata = load_imputeformer_checkpoint(checkpoint, target_device)
    variable = metadata.get("variable")
    if not isinstance(variable, str):
        raise ValueError("Checkpoint contains an invalid variable")
    panel = build_spatiotemporal_panel(data_root, variable)
    case = BenchmarkCase(
        cast(SyntheticMaskType, mode),
        point_rate if mode in {"point", "empirical"} else block_length,
    )
    comparison, _rows = _benchmark_loaded_model(
        model,
        metadata,
        panel,
        split,
        case=case,
        seed=TrainingConfig.from_checkpoint(metadata).seed,
        max_batches=max_batches,
        device=target_device,
    )
    return comparison


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(
        description="Compare ImputeFormer and interpolation on shared masks",
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/2-processed-v2"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/imputeformer"),
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=VARIABLE_TYPE_NAMES,
        default=VARIABLE_TYPE_NAMES,
    )
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        help=(
            "load seed-specific checkpoints and pair each training seed with "
            "the same mask seed"
        ),
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0914
    """Run the declared suite without selecting masks after seeing results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args(argv)
    protocol = protocol_with_overrides(load_benchmark_protocol(args.config), args)
    device = default_device()
    rows: list[BenchmarkRow] = []
    comparisons: list[dict[str, Any]] = []
    checkpoint_protocol: dict[str, Any] = {}
    paired_training = args.training_seeds is not None
    seed_mode = "paired_training_and_mask" if paired_training else "mask_only"
    for variable in args.variables:
        panel = build_spatiotemporal_panel(args.data_dir, variable)
        checkpoint_seeds: tuple[int | None, ...] = (
            tuple(protocol.seeds) if paired_training else (None,)
        )
        for checkpoint_seed in checkpoint_seeds:
            checkpoint = imputeformer_checkpoint_path(
                args.checkpoint_dir,
                variable,
                seed=checkpoint_seed,
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            model, metadata = load_imputeformer_checkpoint(checkpoint, device)
            if metadata.get("variable") != variable:
                raise ValueError(
                    f"Checkpoint metadata does not match path: {checkpoint}"
                )
            checkpoint_config = TrainingConfig.from_checkpoint(metadata)
            if (
                checkpoint_seed is not None
                and checkpoint_config.seed != checkpoint_seed
            ):
                raise ValueError(
                    f"Checkpoint {checkpoint} declares training seed "
                    f"{checkpoint_config.seed}, expected {checkpoint_seed}"
                )
            checkpoint_key = (
                f"seed-{checkpoint_seed}/{variable}"
                if checkpoint_seed is not None
                else variable
            )
            checkpoint_protocol[checkpoint_key] = {
                "path": str(checkpoint),
                "model_config": metadata.get("model_config"),
                "training_config": metadata.get("training_config"),
                "training_objective": metadata.get("training_objective"),
                "training_run": metadata.get("training_run"),
            }
            evaluation_seeds = (
                (checkpoint_seed,) if checkpoint_seed is not None else protocol.seeds
            )
            for seed in evaluation_seeds:
                for case in protocol.cases:
                    comparison, case_rows = _benchmark_loaded_model(
                        model,
                        metadata,
                        panel,
                        protocol.split,
                        case=case,
                        seed=seed,
                        max_batches=protocol.max_batches,
                        device=device,
                    )
                    comparison["training_seed"] = checkpoint_config.seed
                    comparisons.append(comparison)
                    rows.extend(case_rows)
                    logger.info(
                        "%s seed=%d %s: model %.6f, interpolation %.6f, delta %.2f%%",
                        variable,
                        seed,
                        case.label,
                        comparison["model_mae"],
                        comparison["baseline_mae"],
                        comparison["mae_improvement_pct"],
                    )
    protocol_payload = {
        "config": str(args.config),
        "data_dir": str(args.data_dir),
        "split": protocol.split,
        "seeds": list(protocol.seeds),
        "seed_mode": seed_mode,
        "max_batches": protocol.max_batches,
        "cases": [asdict(case) for case in protocol.cases],
        "checkpoints": checkpoint_protocol,
        "code": git_provenance(Path.cwd().resolve()),
    }
    output_path, markdown_output_path = benchmark_artifact_paths(
        "imputeformer",
        args.variables,
        split=protocol.split,
        seeds=protocol.seeds,
        seed_mode=seed_mode,
        output=args.output,
        markdown_output=args.markdown_output,
    )
    payload = write_benchmark_artifact(
        output_path,
        rows,
        protocol=protocol_payload,
        comparisons=comparisons,
    )
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.write_text(markdown_table(rows), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
