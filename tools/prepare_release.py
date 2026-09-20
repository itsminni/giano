"""Verify release artifacts, run the demo and build a local archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import IO, Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import yaml

from giano.provenance import file_sha256, git_provenance, training_dataset_identity
from giano.variables import VARIABLE_TYPE_NAMES

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notebooks.release_artifacts import (  # noqa: E402
    DOWNSTREAM_PATH,
    RELEASE_PATH,
    SEEDS,
    load_benchmarks,
    load_downstream,
    read_completed,
)

PATH_KEYS = (
    "training",
    "checkpoint_report",
    "evaluation",
    "downstream",
    "external_evaluation",
    "onnx",
    "demo",
    "stations",
    "research",
    "history",
    "data",
)


def safe_name(name: str) -> str:
    """Validate a relative archive path."""
    path = PurePosixPath(name)
    if (
        not name
        or name == "."
        or path.is_absolute()
        or path.as_posix() != name
        or any(part in (".", "..") for part in path.parts)
        or "\\" in name
        or ":" in name
        or any(ord(c) < 32 for c in name)
    ):
        raise ValueError(f"Unsafe relative path: {name!r}")
    return name


def local_path(root: Path, name: str) -> Path:
    path = root / safe_name(name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes root: {name}")
    if any(part.is_symlink() for part in (path, *path.parents) if part != root.parent):
        raise ValueError(f"Symlink not permitted: {name}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def profile(root: Path) -> dict[str, Any]:
    result = yaml.safe_load((root / "release.yaml").read_text())
    if result.get("schema_version") != 1 or tuple(result.get("seeds", ())) != SEEDS:
        raise ValueError("Unexpected release schema or training seeds")
    for key in PATH_KEYS:
        local_path(root, result[key])
    if (
        Path(result["evaluation"]) != RELEASE_PATH
        or Path(result["downstream"]) != DOWNSTREAM_PATH
    ):
        raise ValueError("Release readers require the corrected evaluation snapshot")
    if result["presentation"] != {
        "family": "giano",
        "seed": 42,
        "fallback": "none",
        "operational_policy": "deferred",
    }:
        raise ValueError("Unexpected presentation settings")
    return result


def require_hash(path: Path, expected: str) -> None:
    if file_sha256(path) != expected:
        raise ValueError(f"Hash mismatch: {path}")


def verify_inventory(root: Path) -> int:
    manifest = read_json(root / "manifest.json")
    files = manifest["files"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != {*files, "manifest.json"}:
        raise ValueError(f"Unexpected or missing inventory files: {root}")
    for name, record in files.items():
        path = local_path(root, name)
        require_hash(path, record if isinstance(record, str) else record["sha256"])
        if isinstance(record, dict) and path.stat().st_size != record["bytes"]:
            raise ValueError(f"Size mismatch: {path}")
    return len(files)


def check(
    root: Path, *, with_data: bool = False, with_history: bool = False
) -> dict[str, Any]:
    cfg = profile(root)
    training = root / cfg["training"]
    report = read_json(root / cfg["checkpoint_report"])
    require_hash(training / "manifest.json", report["training_manifest_sha256"])
    manifest = read_json(training / "manifest.json")
    require_hash(training / "config.yaml", manifest["inputs"]["sources"]["config.yaml"])
    expected = {
        (family, variable, seed)
        for family in ("giano", "fair", "legacy_retrained")
        for variable in VARIABLE_TYPE_NAMES
        for seed in SEEDS
    }
    rows = report["checkpoints"]
    if (
        len(rows) != 90
        or {(r["family"], r["variable"], r["seed"]) for r in rows} != expected
    ):
        raise ValueError("Incomplete or duplicate checkpoint registry")
    for row in rows:
        require_hash(local_path(root, row["path"]), row["sha256"])
    audit = read_json(root / cfg["evaluation"] / "analysis/model_invariants.json")
    require_hash(root / "src/giano/model/imputeformer.py", audit["model_source_sha256"])
    benchmarks, sources = load_benchmarks(root)
    downstream = {}
    for variable in VARIABLE_TYPE_NAMES:
        read_completed(
            root / cfg["evaluation"] / "detailed" / variable / "validation.json.gz"
        )
        if variable == "wind_speed":
            payload, _ = read_completed(
                root / cfg["downstream"] / "downstream/wind_speed.json",
                postprocessing_key="repair_prediction_postprocessing",
            )
            if (
                payload["records"]
                or payload["protocol"].get("evaluation_status") != "not_evaluable"
                or variable not in payload["protocol"].get("unavailable_variables", {})
            ):
                raise ValueError(
                    "Expected explicit wind-speed downstream unavailability"
                )
            downstream[variable] = {"status": "not_evaluable", "records": 0}
        else:
            payload, _ = load_downstream(root, variable)
            downstream[variable] = {
                "status": "complete",
                "records": len(payload["records"]),
            }
        export = read_json(root / cfg["onnx"] / f"giano_{variable}_seed42.json")
        require_hash(local_path(root, export["onnx"]), export["onnx_sha256"])
        checkpoint = next(
            r
            for r in rows
            if (r["family"], r["variable"], r["seed"]) == ("giano", variable, 42)
        )
        if export["checkpoint_sha256"] != checkpoint["sha256"]:
            raise ValueError(f"ONNX checkpoint binding changed: {variable}")
        if (
            not math.isfinite(export["max_abs_error"])
            or export["max_abs_error"] >= 1e-4
        ):
            raise ValueError(f"ONNX saved parity failed: {variable}")
    inventories = {
        key: verify_inventory(root / cfg[key])
        for key in ("demo", "stations", "research")
    }
    if with_history:
        inventories["history"] = verify_inventory(root / cfg["history"])
    if with_data:
        for variable in VARIABLE_TYPE_NAMES:
            if (
                training_dataset_identity(root / cfg["data"], variable)
                != manifest["inputs"]["datasets"][variable]
            ):
                raise ValueError(f"Training dataset changed: {variable}")
    return {
        "verified_at": datetime.now(UTC).isoformat(),
        "status": "verified",
        "release": cfg["name"],
        "checkpoint_count": len(rows),
        "common_jobs": len(sources),
        "unique_benchmark_rows": len(benchmarks),
        "detailed_jobs": len(VARIABLE_TYPE_NAMES),
        "downstream": downstream,
        "onnx_exports_hash_checked": 6,
        "onnx_parity": "recorded CPU check",
        "inventories": inventories,
        "processed_data_hashes_checked": with_data,
        "history_hashes_checked": with_history,
        "original_campaign_state": read_json(root / cfg["evaluation"] / "status.json")[
            "state"
        ],
        "repaired_downstream_state": read_json(
            root / cfg["downstream"] / "status.json"
        )["state"],
        "limits": [
            "Previously used validation/test periods",
            "Unverified source timing; grid-proxy model coordinates",
            "Natural gaps: unknown truth; unsupported estimates: null",
        ],
    }


def tensor(record: dict[str, Any]) -> np.ndarray:
    if record["dtype"] not in ("float32", "bool"):
        raise ValueError("Unsupported demo tensor dtype")
    shape = record["shape"]
    if not all(type(n) is int and n > 0 for n in shape):
        raise ValueError("Invalid demo tensor shape")
    values = np.asarray(record["values"], dtype=record["dtype"]).reshape(shape)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite demo tensor")
    return values


def demo(root: Path) -> dict[str, Any]:
    """Check PyTorch predictions on two saved inputs."""
    import torch

    from giano.model.train_imputeformer import load_imputeformer_checkpoint

    cfg = profile(root)
    directory = root / cfg["demo"]
    verify_inventory(directory)
    manifest = read_json(directory / "manifest.json")
    registry = read_json(root / cfg["checkpoint_report"])["checkpoints"]
    entry = next(
        r
        for r in registry
        if (r["family"], r["variable"], r["seed"]) == ("giano", "temperature", 42)
    )
    checkpoint = local_path(root, entry["path"])
    require_hash(checkpoint, entry["sha256"])
    if manifest["checkpoint_sha256"] != entry["sha256"]:
        raise ValueError("Demo/checkpoint identity mismatch")
    model, _ = load_imputeformer_checkpoint(checkpoint, torch.device("cpu"))
    results = []
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for example in read_json(directory / "examples.json")["examples"]:
            inputs = example["inputs"]
            with torch.inference_mode():
                outputs = model(
                    *(
                        torch.from_numpy(tensor(inputs[key]))
                        for key in (
                            "features",
                            "coordinates",
                            "node_mask",
                            "baseline",
                        )
                    )
                )
            errors = []
            for name, output in zip(("prediction", "residual"), outputs, strict=True):
                reference = tensor(example["reference_outputs"][name])
                actual = output.numpy()
                if actual.shape != reference.shape or not np.isfinite(actual).all():
                    raise ValueError("Invalid model output")
                errors.append(float(np.abs(actual - reference).max()))
            if max(errors) >= 1e-4:
                raise ValueError(f"Real-weight demo differs from reference: {errors}")
            results.append(
                {"example": example["name"], "normalized_max_abs_error": max(errors)}
            )
    finally:
        torch.set_num_threads(previous_threads)
    if len(results) != 2:
        raise ValueError("Expected two fixed validation examples")
    return {
        "status": "passed",
        "device": "cpu",
        "checkpoint_sha256": entry["sha256"],
        "examples": results,
        "training_started": False,
        "browser_tested": False,
    }


def source_paths(root: Path) -> list[str]:
    result = subprocess.run(  # noqa: S603
        ["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"],  # noqa: S607
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(set(result.stdout.decode().rstrip("\0").split("\0")))


def source_guard(name: str, path: Path) -> None:
    safe_name(name)
    parts = {part.lower() for part in PurePosixPath(name).parts}
    if (
        any(part.startswith(".env") for part in parts)
        or parts & {".git", "credentials.json", "id_rsa", "id_ed25519"}
        or path.suffix.lower() in {".pem", ".key", ".p12"}
    ):
        raise ValueError(f"Potential private file excluded from release: {name}")
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError(f"Unexpectedly large source file; review explicitly: {name}")


def stream_hash(handle: IO[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path) -> dict[str, Any]:
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("Duplicate archive members")
        for info in archive.infolist():
            safe_name(info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Directories/symlinks are not archive payloads")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != 1 or manifest.get("status") != "release":
            raise ValueError("Unexpected release manifest")
        if set(names) != {*manifest["files"], "manifest.json"}:
            raise ValueError("Archive membership differs from manifest")
        for name, record in manifest["files"].items():
            if archive.getinfo(name).file_size != record["bytes"]:
                raise ValueError(f"Archive size mismatch: {name}")
            with archive.open(name) as handle:
                if stream_hash(handle) != record["sha256"]:
                    raise ValueError(f"Archive hash mismatch: {name}")
    return {
        "status": "verified",
        "files": len(names),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def bundle(
    root: Path,
    output: Path,
    *,
    wiki_root: Path | None = None,
    include_data: bool = False,
    include_history: bool = False,
    artifacts_only: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Preserve existing release: {output}")
    cfg = profile(root)
    if artifacts_only and wiki_root is not None:
        raise ValueError("An artifacts-only archive cannot include the wiki")
    prefix = "" if artifacts_only else "project/"
    files: dict[str, Path] = {}
    for name in [] if artifacts_only else source_paths(root):
        path = local_path(root, name)
        if not path.is_file():
            continue
        source_guard(name, path)
        files[f"project/{name}"] = path
    report = read_json(root / cfg["checkpoint_report"])
    for row in report["checkpoints"]:
        files[f"{prefix}{row['path']}"] = local_path(root, row["path"])
    directories = [
        cfg[key]
        for key in (
            "training",
            "evaluation",
            "downstream",
            "external_evaluation",
            "onnx",
            "demo",
            "stations",
            "research",
        )
    ]
    if include_history:
        directories.append(cfg["history"])
    if include_data:
        directories.append(cfg["data"])
    for name in directories:
        for path in (root / name).rglob("*"):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                files[f"{prefix}{relative}"] = local_path(root, relative)
    if wiki_root is not None:
        for path in sorted(wiki_root.glob("*.md")):
            source_guard(path.name, path)
            files[f"wiki/{path.name}"] = local_path(wiki_root, path.name)
        if not any(name.startswith("wiki/") for name in files):
            raise ValueError("Requested wiki has no Markdown pages")
    # Hash before verification to detect edits during packaging.
    records = {
        name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        for name, path in sorted(files.items())
    }
    verification = check(root, with_data=include_data, with_history=include_history)
    real_demo = demo(root)
    start = (
        (
            b"# Giano 1.0.0 artifacts\n\n"
            b"Extract checkpoints/ and artifacts/ into the cloned Giano repository.\n"
            b"Dataset: https://drive.google.com/drive/folders/1DPsYf6dVRtCqxE3mDYAx4evTj0WASXK7\n"
            b"Checksums: manifest.json. Verification: python tools/prepare_release.py check --with-history\n"
        )
        if artifacts_only
        else (
            b"# Giano release archive\n\n"
            b"Extract into a new directory and follow project/README.md.\n"
            b"Artifact paths: project/release.yaml. Checksums and release status: manifest.json.\n"
        )
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "release",
        "release": cfg["name"],
        "created_at": datetime.now(UTC).isoformat(),
        "git": git_provenance(root),
        "wiki_git": git_provenance(wiki_root) if wiki_root else None,
        "includes": {
            "processed_data": include_data,
            "full_history": include_history,
            "wiki": wiki_root is not None,
            "source": not artifacts_only,
        },
        "verification": verification,
        "real_weight_demo": real_demo,
        "files": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="giano-release-", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "release.zip"
        with ZipFile(staged, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name, path in sorted(files.items()):
                safe_name(name)
                archive.write(path, name)
                manifest["files"][name] = records[name]
            archive.writestr("START_HERE.md", start)
            manifest["files"]["START_HERE.md"] = {
                "sha256": hashlib.sha256(start).hexdigest(),
                "bytes": len(start),
            }
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, allow_nan=False),
            )
        result = verify_archive(staged)
        # Atomic placement without overwriting an existing archive.
        output.hardlink_to(staged)
    return {**result, "path": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("check", help="Verify release artifacts")
    verify.add_argument("--with-data", action="store_true")
    verify.add_argument("--with-history", action="store_true")
    commands.add_parser("demo", help="Run two saved examples on CPU")
    package = commands.add_parser("bundle", help="Build a local release archive")
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--wiki-root", type=Path)
    package.add_argument("--include-data", action="store_true")
    package.add_argument("--include-history", action="store_true")
    package.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Package weights and results for an existing checkout",
    )
    archive = commands.add_parser(
        "verify-bundle", help="Check ZIP hashes without extracting"
    )
    archive.add_argument("archive", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "check":
        result = check(root, with_data=args.with_data, with_history=args.with_history)
    elif args.command == "demo":
        result = demo(root)
    elif args.command == "bundle":
        result = bundle(
            root,
            args.output.resolve(),
            wiki_root=args.wiki_root.resolve() if args.wiki_root else None,
            include_data=args.include_data,
            include_history=args.include_history,
            artifacts_only=args.artifacts_only,
        )
    else:
        result = verify_archive(args.archive)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
