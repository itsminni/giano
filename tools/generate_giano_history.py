"""Reconstruct station histories, resuming completed variables."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from giano.impute import impute_variable
from giano.provenance import (
    file_sha256,
    training_dataset_identity,
    validate_training_dataset,
)
from giano.variables import VARIABLE_TYPE_NAMES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("artifacts/imputed/giano_history")
LOGGER = logging.getLogger(__name__)


def request_identity(
    data: Path, checkpoint: Path, variable: str, batch_size: int
) -> dict[str, Any]:
    dataset = training_dataset_identity(data, variable)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_training_dataset(raw, dataset)
    if (
        raw.get("model_class") != "MeteorologicalImputeFormer"
        or raw.get("variable") != variable
    ):
        raise ValueError("Wrong Giano checkpoint")
    return {
        "recipe": "giano-natural-history-v1",
        "variable": variable,
        "dataset": dataset,
        "checkpoint_sha256": file_sha256(checkpoint),
        "seed": raw["training_config"]["seed"],
        "fallback": "none",
        "batch_size": batch_size,
        "device": "cpu",
        "torch_threads": 1,
        "torch_version": str(torch.__version__),
        "inference_sources": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in sorted((ROOT / "src/giano").rglob("*.py"))
        },
    }


def verify_completed(
    directory: Path, expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    record = json.loads((directory / "complete.json").read_text())
    if record.get("status") != "complete" or (
        expected is not None and record["request"] != expected
    ):
        raise ValueError(
            f"Changed inference request; preserve {directory} and choose a new output"
        )
    actual = {p.name for p in directory.iterdir() if p.is_file()}
    if actual != set(record["files"]) | {"complete.json"}:
        raise ValueError(f"Missing or unlisted inference outputs: {directory}")
    for name, digest in record["files"].items():
        if Path(name).name != name or file_sha256(directory / name) != digest:
            raise ValueError(f"Changed inference output: {name}")
    return record


def generate(
    data: Path, checkpoint: Path, output: Path, variable: str, batch_size: int = 16
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    identity = request_identity(data, checkpoint, variable, batch_size)
    destination = output / variable
    if destination.exists():
        record = verify_completed(destination, identity)
        LOGGER.info("%s: verified completed output; skipping inference", variable)
        return record
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix=f".{variable}-", dir=output) as temporary:
        staging = Path(temporary) / "completed"
        impute_variable(
            data,
            checkpoint,
            staging,
            batch_size=batch_size,
            device=torch.device("cpu"),
            variable=variable,
            fallback="none",
        )
        if request_identity(data, checkpoint, variable, batch_size) != identity:
            raise ValueError("Inputs changed during inference")
        record = {
            "status": "complete",
            "request": identity,
            "completed_at": datetime.now(UTC).isoformat(),
            "files": {
                p.name: file_sha256(p) for p in sorted(staging.iterdir()) if p.is_file()
            },
        }
        (staging / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
        verify_completed(staging, identity)
        staging.rename(destination)
    LOGGER.info(
        "%s: completed and verified %d station histories",
        variable,
        len(identity["dataset"]["files"]),
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/2-processed-v2"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/corrected_v2_spectral/imputeformer/seed-42"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=VARIABLE_TYPE_NAMES,
        default=VARIABLE_TYPE_NAMES,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    for variable in args.variables:
        generate(
            args.data_dir,
            args.checkpoint_dir / f"{variable}.pt",
            args.output,
            variable,
            args.batch_size,
        )


if __name__ == "__main__":
    main()
