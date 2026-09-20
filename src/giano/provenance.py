"""Small reproducibility helpers shared by experiment artifact writers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from giano.variables import VARIABLE_TYPE_NAMES


def file_sha256(path: Path) -> str:
    """Hash file bytes incrementally rather than relying on path or mtime."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_dataset_identity(data_root: Path, variable: str) -> dict[str, Any]:
    """Identify the exact per-variable files and their model-loading order."""

    if variable not in VARIABLE_TYPE_NAMES:
        raise ValueError(f"Unsupported dataset variable: {variable}")
    paths = sorted(data_root.glob(f"*/*_{variable}_merged.nc"))
    if not paths:
        raise FileNotFoundError(f"No processed files for {variable} in {data_root}")
    if len({path.name.split("_", 1)[0] for path in paths}) != len(paths):
        raise ValueError(f"Duplicate processed station for {variable}")
    files = [
        {"path": path.relative_to(data_root).as_posix(), "sha256": file_sha256(path)}
        for path in paths
    ]
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "algorithm": "sha256-relative-files-v1",
        "variable": variable,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def validate_training_dataset(raw: dict[str, Any], expected: dict[str, Any]) -> None:
    """Refuse resume/reuse when data changed or old provenance is absent."""
    if raw.get("training_dataset") != expected:
        raise ValueError(
            "Checkpoint training dataset identity is missing or has changed; "
            "preserve this checkpoint and use a new output directory"
        )


def git_provenance(project_root: Path) -> dict[str, Any]:
    """Return the current Git commit and whether tracked/untracked files differ."""

    def run(*arguments: str) -> str | None:
        result = subprocess.run(  # noqa: S603
            ("git", *arguments),  # noqa: S607
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }
