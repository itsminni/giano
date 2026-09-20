"""Read one verified release snapshot; never mix historical experiments."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from giano.evaluation.gapfill.cases import load_benchmark_protocol
from giano.prediction import prediction_policy
from giano.provenance import file_sha256
from giano.variables import VARIABLE_TYPE_NAMES

SEEDS = (42, 43, 44, 45, 46)
RELEASE_PATH = Path("artifacts/evaluation/corrected_v2_release")
DOWNSTREAM_PATH = Path("artifacts/evaluation/corrected_v2_downstream_v2")


def read_completed(
    path: Path, *, postprocessing_key: str = "prediction_postprocessing"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require the campaign completion marker, not merely a partial file."""
    marker = path.with_name(path.name + ".complete.json")
    if not path.is_file() or not marker.is_file():
        raise FileNotFoundError(f"Release job not complete yet: {path}")
    completed = json.loads(marker.read_text())
    if file_sha256(path) != completed.get("output_sha256"):
        raise ValueError(f"Release output changed after completion: {path}")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text())
    if payload["protocol"].get(postprocessing_key) != prediction_policy():
        raise ValueError(f"Release uses different prediction postprocessing: {path}")
    return payload, completed["request"]["inputs"]


def load_benchmarks(project_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load exactly six variables/five seeds/13 cases and deduplicate references."""
    protocol = load_benchmark_protocol(
        project_root / "artifacts/training/corrected_v2_spectral/config.yaml"
    )
    cases = {(case.mask_type, case.parameter) for case in protocol.cases}
    expected = {
        (variable, mask, parameter, seed, family)
        for variable in VARIABLE_TYPE_NAMES
        for mask, parameter in cases
        for seed in SEEDS
        for family in (
            "giano/imputeformer",
            "bilstm/fair",
            "bilstm/legacy_retrained",
            "interpolation",
        )
    }
    indexed: dict[tuple[str, str, int | float, int, str], dict[str, Any]] = {}
    sources = []
    identities = set()
    for variable in VARIABLE_TYPE_NAMES:
        directory = (
            project_root
            / RELEASE_PATH
            / "validation"
            / variable
            / "val_paired-seeds-42-43-44-45-46"
        )
        for family in ("imputeformer", "bilstm"):
            path = directory / f"{family}_benchmark.json"
            payload, identity = read_completed(path)
            identities.add(json.dumps(identity, sort_keys=True))
            sources.append(str(path))
            metadata = payload["protocol"]
            if (
                payload.get("schema_version") != 2
                or metadata.get("split") != "val"
                or tuple(metadata.get("seeds", ())) != SEEDS
                or metadata.get("seed_mode") != "paired_training_and_mask"
            ):
                raise ValueError(f"Not the paired release validation protocol: {path}")
            for row in payload["results"]:
                label = (
                    "interpolation"
                    if row["model"] == "interpolation"
                    else f"{row['model']}/{row['model_variant']}"
                )
                key = (
                    row["variable"],
                    row["mask_type"],
                    row["mask_parameter"],
                    row["seed"],
                    label,
                )
                if row["variable"] != variable or row["split"] != "val":
                    raise ValueError(f"Unexpected variable/split in {path}")
                if (
                    not np.isfinite([row["mae"], row["rmse"]]).all()
                    or row["n_hidden"] <= 0
                ):
                    raise ValueError(f"Invalid release score in {path}")
                if key in indexed:
                    previous = indexed[key]
                    # Runtime differs, but duplicated references must score identically.
                    if label != "interpolation" or any(
                        previous[name] != row[name]
                        for name in ("mae", "rmse", "n_hidden")
                    ):
                        raise ValueError(
                            f"Conflicting or duplicate release rows: {key}"
                        )
                    continue
                indexed[key] = row
    if len(identities) != 1 or set(indexed) != expected:
        raise ValueError("Incomplete or mixed release input snapshots")
    for variable, mask, parameter, seed, _ in expected:
        counts = {
            indexed[(variable, mask, parameter, seed, family)]["n_hidden"]
            for family in (
                "giano/imputeformer",
                "bilstm/fair",
                "bilstm/legacy_retrained",
                "interpolation",
            )
        }
        if len(counts) != 1:
            raise ValueError("Release models did not score the same hidden counts")
    return list(indexed.values()), sources


def load_downstream(project_root: Path, variable: str) -> tuple[dict[str, Any], Path]:
    """Load the explicitly chosen variable."""
    if variable not in VARIABLE_TYPE_NAMES:
        raise ValueError(f"Unknown variable: {variable}")
    path = project_root / DOWNSTREAM_PATH / "downstream" / f"{variable}.json"
    payload, _ = read_completed(
        path, postprocessing_key="repair_prediction_postprocessing"
    )
    if variable in payload["protocol"].get("unavailable_variables", {}):
        reason = payload["protocol"]["unavailable_variables"][variable]["reason"]
        raise ValueError(f"Downstream not evaluable for {variable}: {reason}")
    if payload.get("schema_version") != 2 or not payload.get("records"):
        raise ValueError("Expected nonempty downstream schema 2")
    if {row["variable"] for row in payload["records"]} != {variable}:
        raise ValueError("Downstream artifact contains an unexpected variable")
    return payload, path
