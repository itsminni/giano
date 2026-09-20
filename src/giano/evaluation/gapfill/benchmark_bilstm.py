"""Evaluate both BiLSTM recipes with the common gap-filling benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from giano.baselines.bilstm import BiLSTMVariant
from giano.baselines.train_bilstm import (
    BiLSTMTrainingConfig,
    _loader,
    load_bilstm_checkpoint,
)
from giano.evaluation.gapfill.cases import (
    dataset_for_case,
    load_benchmark_protocol,
    protocol_with_overrides,
)
from giano.evaluation.gapfill.results import (
    BenchmarkRow,
    benchmark_artifact_paths,
    comparison_payload,
    evaluate_model,
    markdown_table,
    rows_from_comparison,
    write_benchmark_artifact,
)
from giano.experiments import bilstm_checkpoint_path
from giano.provenance import git_provenance
from giano.runtime import default_device
from giano.spatiotemporal_dataset import (
    build_spatiotemporal_panel,
)
from giano.variables import VARIABLE_TYPE_NAMES

logger = logging.getLogger(__name__)


def _checkpoint_training_config(raw: dict[str, Any]) -> BiLSTMTrainingConfig:
    values = raw.get("training_config")
    if not isinstance(values, dict):
        raise ValueError("BiLSTM checkpoint lacks training_config")
    normalized = dict(values)
    normalized["block_lengths"] = tuple(normalized.get("block_lengths", (3, 6, 12, 24)))
    config = BiLSTMTrainingConfig(**normalized)
    config.validate()
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse BiLSTM benchmark arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark legacy-retrained and fair BiLSTMs on shared masks"
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/2-processed-v2"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/bilstm")
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=VARIABLE_TYPE_NAMES,
        default=("temperature",),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("legacy_retrained", "fair"),
        default=("legacy_retrained", "fair"),
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
    """Evaluate every requested checkpoint without rematerializing protocols."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args(argv)
    protocol = protocol_with_overrides(load_benchmark_protocol(args.config), args)
    device = default_device()
    rows: list[BenchmarkRow] = []
    comparisons: list[dict[str, Any]] = []
    interpolation_keys: set[tuple[str, int, str, int | float]] = set()
    checkpoint_protocol: dict[str, Any] = {}
    paired_training = args.training_seeds is not None
    seed_mode = "paired_training_and_mask" if paired_training else "mask_only"
    panels = {
        variable: build_spatiotemporal_panel(args.data_dir, variable)
        for variable in args.variables
    }
    for variant_raw in args.variants:
        variant = cast(BiLSTMVariant, variant_raw)
        for variable in args.variables:
            checkpoint_seeds: tuple[int | None, ...] = (
                tuple(protocol.seeds) if paired_training else (None,)
            )
            for checkpoint_seed in checkpoint_seeds:
                checkpoint = bilstm_checkpoint_path(
                    args.checkpoint_dir,
                    variant,
                    variable,
                    seed=checkpoint_seed,
                )
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
                model, metadata = load_bilstm_checkpoint(checkpoint, device)
                if (
                    metadata.get("variant") != variant
                    or metadata.get("variable") != variable
                ):
                    raise ValueError(
                        f"Checkpoint metadata does not match path: {checkpoint}"
                    )
                base_config = _checkpoint_training_config(metadata)
                if checkpoint_seed is not None and base_config.seed != checkpoint_seed:
                    raise ValueError(
                        f"Checkpoint {checkpoint} declares training seed "
                        f"{base_config.seed}, expected {checkpoint_seed}"
                    )
                checkpoint_key = (
                    f"{variant}/seed-{checkpoint_seed}/{variable}"
                    if checkpoint_seed is not None
                    else f"{variant}/{variable}"
                )
                checkpoint_protocol[checkpoint_key] = {
                    "path": str(checkpoint),
                    "model_config": metadata.get("model_config"),
                    "training_config": metadata.get("training_config"),
                    "training_run": metadata.get("training_run"),
                    "recipe": metadata.get("recipe"),
                }
                evaluation_seeds = (
                    (checkpoint_seed,)
                    if checkpoint_seed is not None
                    else protocol.seeds
                )
                for seed in evaluation_seeds:
                    config = replace(base_config, seed=seed)
                    for case in protocol.cases:
                        dataset = dataset_for_case(
                            panels[variable], protocol.split, config, case, seed
                        )
                        started = time.perf_counter()
                        result = evaluate_model(
                            model,
                            _loader(dataset, config, shuffle=False),
                            device,
                            variable,
                            max_batches=protocol.max_batches,
                        )
                        runtime_ms = (time.perf_counter() - started) * 1000.0
                        comparison = comparison_payload(
                            result,
                            variable=variable,
                            split=protocol.split,
                            seed=seed,
                            case=case,
                            runtime_ms=runtime_ms,
                        )
                        comparison["model_variant"] = variant
                        comparison["training_seed"] = base_config.seed
                        comparisons.append(comparison)
                        model_row, interpolation_row = rows_from_comparison(
                            comparison,
                            model="bilstm",
                            model_variant=variant,
                            variable=variable,
                            seed=seed,
                            split=protocol.split,
                            case=case,
                            runtime_ms=runtime_ms,
                            parameter_count=sum(
                                parameter.numel() for parameter in model.parameters()
                            ),
                        )
                        rows.append(model_row)
                        interpolation_key = (
                            variable,
                            seed,
                            case.mask_type,
                            case.parameter,
                        )
                        if interpolation_key not in interpolation_keys:
                            rows.append(interpolation_row)
                            interpolation_keys.add(interpolation_key)
                        logger.info(
                            "%s/%s seed=%d %s: MAE %.6f, interpolation %.6f",
                            variant,
                            variable,
                            seed,
                            case.label,
                            result.model_mae,
                            result.baseline_mae,
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
        "bilstm",
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
