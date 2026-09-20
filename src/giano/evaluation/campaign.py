"""Run corrected-data release evaluations sequentially.

Invoke from the checkout with ``python -m giano.evaluation.campaign``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from giano.evaluation.forecasting.run import main as downstream_main
from giano.evaluation.gapfill.benchmark_bilstm import main as bilstm_main
from giano.evaluation.gapfill.benchmark_imputeformer import main as giano_main
from giano.evaluation.gapfill.detailed import main as detailed_main
from giano.provenance import file_sha256, training_dataset_identity
from giano.training_campaign import (
    SEEDS,
    Stage,
    _campaign_lock,
    _interrupt,
    _run_stages,
    _write_json,
)

VARIABLES = (
    "temperature",
    "humidity",
    "pressure",
    "precipitation",
    "wind_speed",
    "wind_direction",
)
ARTIFACT_ROOT = Path("artifacts/evaluation/corrected_v2_release")
DOWNSTREAM_ROOT = Path("artifacts/evaluation/corrected_v2_downstream_v2")


def _jobs(root: Path, *, phase: str = "all") -> list[tuple[Stage, Path]]:
    destination = root / (ARTIFACT_ROOT if phase == "all" else DOWNSTREAM_ROOT)
    config = root / "artifacts/training/corrected_v2_spectral/config.yaml"
    data = root / "data/2-processed-v2"
    giano = root / "checkpoints/corrected_v2_spectral/imputeformer"
    bilstm = root / "checkpoints/corrected_v2/bilstm"
    common = ["--config", str(config), "--data-dir", str(data)]
    seeds = [str(seed) for seed in SEEDS]
    jobs: list[tuple[Stage, Path]] = []
    for variable in VARIABLES:
        directory = (
            destination
            / "validation"
            / variable
            / ("val_paired-seeds-" + "-".join(seeds))
        )
        for family, trainer, checkpoints in (
            ("giano", giano_main, giano),
            ("bilstm", bilstm_main, bilstm),
        ):
            filename = "imputeformer" if family == "giano" else family
            output = directory / f"{filename}_benchmark.json"
            arguments = [
                *common,
                "--variables",
                variable,
                "--split",
                "val",
                "--training-seeds",
                *seeds,
                "--max-batches",
                "200",
                "--checkpoint-dir",
                str(checkpoints),
                "--output",
                str(output),
            ]
            jobs.append(
                ((f"validation/{variable}/{family}", trainer, arguments), output)
            )
    for variable in VARIABLES:
        output = destination / "detailed" / variable / "validation.json.gz"
        arguments = [
            *common,
            "--variables",
            variable,
            "--split",
            "val",
            "--seeds",
            *seeds,
            "--max-batches",
            "200",
            "--bootstrap-samples",
            "2000",
            "--imputeformer-checkpoint-dir",
            str(giano),
            "--bilstm-checkpoint-dir",
            str(bilstm),
            "--output",
            str(output),
        ]
        jobs.append(((f"detailed/{variable}", detailed_main, arguments), output))
    for variable in VARIABLES:
        output = destination / "downstream" / f"{variable}.json"
        arguments = [
            *common,
            "--variables",
            variable,
            "--seeds",
            *seeds,
            "--max-origins",
            "200",
            "--record-unavailable",
            "--imputeformer-checkpoint-dir",
            str(giano),
            "--bilstm-checkpoint-dir",
            str(bilstm),
            "--output",
            str(output),
        ]
        jobs.append(((f"downstream/{variable}", downstream_main, arguments), output))
    return (
        jobs
        if phase == "all"
        else [job for job in jobs if job[0][0].startswith("downstream/")]
    )


def _input_identity(root: Path) -> dict[str, Any]:
    training = json.loads(
        (root / "artifacts/training/corrected_v2_spectral/manifest.json").read_text()
    )
    datasets = {
        variable: training_dataset_identity(root / "data/2-processed-v2", variable)
        for variable in VARIABLES
    }
    if datasets != training["inputs"]["datasets"]:
        raise ValueError("Evaluation data differ from the completed training campaign")
    checkpoints = {}
    for directory in (
        "checkpoints/corrected_v2_spectral/imputeformer",
        "checkpoints/corrected_v2/bilstm/fair",
        "checkpoints/corrected_v2/bilstm/legacy_retrained",
    ):
        for variable in VARIABLES:
            for seed in SEEDS:
                path = root / directory / f"seed-{seed}/{variable}.pt"
                checkpoints[str(path.relative_to(root))] = file_sha256(path)
    excluded = {
        "impute.py",
        "training_campaign.py",
        "evaluation/campaign.py",
    }
    source = root / "src/giano"
    code = {
        str(path.relative_to(source)): file_sha256(path)
        for path in sorted(source.rglob("*.py"))
        if str(path.relative_to(source)) not in excluded
    }
    return {
        "datasets": datasets,
        "checkpoints": checkpoints,
        "code": code,
        "config_sha256": file_sha256(
            root / "artifacts/training/corrected_v2_spectral/config.yaml"
        ),
        "lock_sha256": file_sha256(root / "uv.lock"),
    }


def _cached_job(stage: Stage, output: Path, identity: dict[str, Any]) -> Stage:
    name, trainer, arguments = stage
    marker = output.with_name(output.name + ".complete.json")
    expected = {"arguments": arguments, "inputs": identity}

    def execute(_arguments: list[str]) -> int:
        if marker.exists():
            completed = json.loads(marker.read_text())
            if (
                completed.get("request") != expected
                or not output.exists()
                or completed.get("output_sha256") != file_sha256(output)
            ):
                raise ValueError(
                    f"Completed evaluation has changed inputs or output: {name}"
                )
            logging.info("Skipping verified completed evaluation %s", name)
            return 0
        if output.exists():
            raise FileExistsError(
                f"Unverified evaluation output exists: {output}; preserve and inspect it before resuming"
            )
        code = trainer(arguments)
        if code == 0:
            _write_json(
                marker, {"request": expected, "output_sha256": file_sha256(output)}
            )
        return code

    return name, execute, arguments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--phase",
        choices=("all", "downstream"),
        default="all",
        help="Downstream uses a separate provenance snapshot and output directory",
    )
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    directory = root / (ARTIFACT_ROOT if args.phase == "all" else DOWNSTREAM_ROOT)
    directory.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    with _campaign_lock(directory / "campaign.lock"):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(directory / "campaign.log"),
                logging.StreamHandler(),
            ],
        )
        identity = _input_identity(root)
        stages = [
            _cached_job(stage, output, identity)
            for stage, output in _jobs(root, phase=args.phase)
        ]
        _write_json(
            directory / "manifest.json",
            {
                "pid": os.getpid(),
                "phase": args.phase,
                "inputs": identity,
                "jobs": [{"name": stage[0], "arguments": stage[2]} for stage in stages],
            },
        )
        awake = None
        if sys.platform == "darwin":
            awake = subprocess.Popen(  # noqa: S603
                ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())]
            )
        try:
            return _run_stages(stages, directory / "status.json", lambda: None)
        finally:
            if awake is not None:
                awake.terminate()
                awake.wait()


if __name__ == "__main__":
    raise SystemExit(main())
