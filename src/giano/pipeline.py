"""Run preprocessing, ImputeFormer training, benchmarking, and imputation."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from giano.variables import VARIABLE_TYPE_NAMES

PROJECT_ROOT = Path.cwd().resolve()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _run_step(step_name: str, command: list[str]) -> None:
    """Run one fixed-module subprocess and fail fast."""
    logger.info("Running %s: %s", step_name, " ".join(command))
    result = subprocess.run(  # noqa: S603
        command,
        cwd=PROJECT_ROOT,
        check=False,
        shell=False,
    )
    if result.returncode:
        raise RuntimeError(f"{step_name} failed with exit code {result.returncode}")


def _config_path(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def _processed_path(config_path: Path) -> Path:
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration top-level must be a mapping")
    paths = raw.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("Configuration paths must be a mapping")
    processed = Path(str(paths.get("processed", "data/2-processed-v2"))).expanduser()
    return (
        processed if processed.is_absolute() else (PROJECT_ROOT / processed).resolve()
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse full-pipeline arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Giano: preprocess, train ImputeFormer, benchmark, and write "
            "non-destructive imputations"
        ),
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=VARIABLE_TYPE_NAMES,
        default=VARIABLE_TYPE_NAMES,
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run a training smoke test"
    )
    parser.add_argument("--skip-unify", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-imputation", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute the maintained pipeline in order."""
    args = parse_args(argv)
    if args.skip_unify != args.skip_split:
        logger.error("Unification and split must be skipped together")
        return 2
    if all(
        (
            args.skip_unify,
            args.skip_split,
            args.skip_train,
            args.skip_benchmark,
            args.skip_imputation,
        ),
    ):
        logger.error("Nothing to run: every stage is skipped")
        return 2
    try:
        config_path = _config_path(args.config)
        processed_path = _processed_path(config_path)
    except (OSError, TypeError, ValueError):
        logger.exception("Invalid configuration")
        return 2

    python = sys.executable
    variables = [str(variable) for variable in args.variables]
    checkpoint_dir = PROJECT_ROOT / "checkpoints/imputeformer"
    train_command = [
        python,
        "-m",
        "giano.model.train_imputeformer",
        "--config",
        str(config_path),
        "--data-dir",
        str(processed_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--variables",
        *variables,
    ]
    if args.quick:
        train_command.append("--quick")
    benchmark_command = [
        python,
        "-m",
        "giano.evaluation.gapfill.benchmark_imputeformer",
        "--config",
        str(config_path),
        "--data-dir",
        str(processed_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--variables",
        *variables,
    ]
    imputation_command = [
        python,
        "-m",
        "giano.impute",
        "--data-dir",
        str(processed_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--variables",
        *variables,
    ]

    try:
        if not args.skip_unify:
            with tempfile.TemporaryDirectory(prefix="giano-merged-") as merged_dir:
                _run_step(
                    "dataset unification",
                    [
                        python,
                        "-m",
                        "giano.data_processing.unify_dataset",
                        "--config",
                        str(config_path),
                        "--out-dir",
                        merged_dir,
                    ],
                )
                _run_step(
                    "dataset publication",
                    [
                        python,
                        "-m",
                        "giano.data_processing.split_dataset",
                        "--config",
                        str(config_path),
                        "--source-folder",
                        merged_dir,
                        "--output-folder",
                        str(processed_path),
                    ],
                )
        if not args.skip_train:
            _run_step("ImputeFormer training", train_command)
        if not args.skip_benchmark:
            _run_step("same-mask benchmark", benchmark_command)
        if not args.skip_imputation:
            _run_step("natural-gap imputation", imputation_command)
    except KeyboardInterrupt:
        return 130
    except RuntimeError:
        logger.exception("Pipeline execution failed")
        return 1
    logger.info("Full Giano pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
