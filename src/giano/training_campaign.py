"""Run the fixed, resumable corrected-data campaign on one local GPU.

Checkout launcher for Windows, macOS and Linux.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from filelock import FileLock, Timeout

from giano.baselines.train_bilstm import main as bilstm_main
from giano.model.train_imputeformer import main as giano_main
from giano.model.train_imputeformer import training_objective
from giano.provenance import file_sha256, git_provenance, training_dataset_identity

logger = logging.getLogger(__name__)
SEEDS = (42, 43, 44, 45, 46)
VARIABLES = (
    "humidity",
    "precipitation",
    "temperature",
    "pressure",
    "wind_speed",
    "wind_direction",
)
ARTIFACT_DIRECTORY = Path("artifacts/training/corrected_v2_spectral")
Stage = tuple[str, Callable[[list[str]], int], list[str]]


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@contextmanager
def _campaign_lock(path: Path) -> Iterator[None]:
    lock = FileLock(path, timeout=0, preserve_lock_file=True, fallback_to_soft=False)
    try:
        lock.acquire()
    except Timeout as error:
        raise RuntimeError("This training campaign is already running") from error
    try:
        yield
    finally:
        lock.release()


def _source_identity(root: Path) -> dict[str, str]:
    paths = sorted((root / "src/giano").rglob("*.py"))
    if not paths:
        raise FileNotFoundError("Run this launcher from the Giano project checkout")
    paths.extend(root / name for name in ("config.yaml", "uv.lock", "pyproject.toml"))
    return {str(path.relative_to(root)): file_sha256(path) for path in paths}


def _freeze_manifest(path: Path, inputs: dict[str, Any], root: Path) -> None:
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("inputs") != inputs:
            raise ValueError(
                "Campaign inputs changed (code, config, environment or data); "
                "do not mix this campaign with a different recipe"
            )
    else:
        _write_json(
            path,
            {"created_at": _timestamp(), "git": git_provenance(root), "inputs": inputs},
        )


def _stages(root: Path, frozen_config: Path) -> list[Stage]:
    common = [
        "--config",
        str(frozen_config),
        "--data-dir",
        str(root / "data/2-processed-v2"),
        "--variables",
        *VARIABLES,
        "--seeds",
        *(str(seed) for seed in SEEDS),
        "--progress-every",
        "100",
        "--resume",
    ]
    return [
        (
            "giano",
            giano_main,
            [
                *common,
                "--checkpoint-dir",
                str(root / "checkpoints/corrected_v2_spectral/imputeformer"),
            ],
        ),
        *[
            (
                variant,
                bilstm_main,
                [
                    *common,
                    "--checkpoint-dir",
                    str(root / "checkpoints/corrected_v2/bilstm"),
                    "--variants",
                    variant,
                ],
            )
            for variant in ("fair", "legacy_retrained")
        ],
    ]


def _run_stages(
    stages: list[Stage], status_path: Path, verify_sources: Callable[[], None]
) -> int:
    status: dict[str, Any] = {
        "pid": os.getpid(),
        "started_at": _timestamp(),
        "stages": {},
    }
    try:
        for name, trainer, arguments in stages:
            status.update(state="running", stage=name, updated_at=_timestamp())
            _write_json(status_path, status)
            verify_sources()
            logger.info("Starting/resuming campaign stage %s", name)
            code = trainer(arguments)
            status["stages"][name] = code
            if code:
                status.update(
                    state="interrupted" if code == 130 else "failed", exit_code=code
                )
                return code
        status.update(state="complete", exit_code=0)
        logger.info("All campaign stages completed successfully")
        return 0
    except KeyboardInterrupt:
        status.update(state="interrupted", exit_code=130)
        logger.warning("Campaign interrupted; rerun the same launcher to resume")
        return 130
    except Exception:
        status.update(state="failed", exit_code=1)
        logger.exception("Campaign failed; later stages will not run")
        return 1
    finally:
        status["updated_at"] = _timestamp()
        _write_json(status_path, status)


def _interrupt(_signal: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    """Freeze provenance, hold one queue lock and execute all three families."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    root = parser.parse_args(argv).project_root.resolve()
    directory = root / ARTIFACT_DIRECTORY
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
        logger.info(
            "Campaign PID %d; six variables, five seeds, three families", os.getpid()
        )
        sources = _source_identity(root)
        inputs: dict[str, Any] = {
            "sources": sources,
            "python": sys.version,
            "torch": str(torch.__version__),
            "training_objective": training_objective(),
            "datasets": {
                variable: training_dataset_identity(
                    root / "data/2-processed-v2", variable
                )
                for variable in VARIABLES
            },
            "stages": [
                {"name": name, "arguments": arguments}
                for name, _, arguments in _stages(root, directory / "config.yaml")
            ],
        }
        _freeze_manifest(directory / "manifest.json", inputs, root)
        frozen_config = directory / "config.yaml"
        if frozen_config.exists():
            if file_sha256(frozen_config) != sources["config.yaml"]:
                raise ValueError("Frozen campaign configuration was modified")
        else:
            frozen_config.write_bytes((root / "config.yaml").read_bytes())

        def verify_sources() -> None:
            if _source_identity(root) != sources:
                raise ValueError("Source/config files changed during the campaign")
            if file_sha256(frozen_config) != sources["config.yaml"]:
                raise ValueError("Frozen campaign configuration was modified")
            for variable in VARIABLES:
                current = training_dataset_identity(
                    root / "data/2-processed-v2", variable
                )
                if current != inputs["datasets"][variable]:
                    raise ValueError(f"Dataset changed during the campaign: {variable}")

        awake = None
        if sys.platform == "darwin":
            awake = subprocess.Popen(  # noqa: S603
                ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())]
            )
        try:
            return _run_stages(
                _stages(root, frozen_config), directory / "status.json", verify_sources
            )
        finally:
            if awake is not None:
                awake.terminate()
                awake.wait()


if __name__ == "__main__":
    raise SystemExit(main())
