"""Summarize controlled validation of unchanged weights on two data versions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from giano.provenance import file_sha256
from giano.variables import VARIABLE_TYPE_NAMES

MODELS = ("giano/imputeformer", "bilstm/fair", "bilstm/legacy_retrained")


def summarize(
    root: Path, *, variables: tuple[str, ...] = tuple(VARIABLE_TYPE_NAMES)
) -> dict[str, Any]:
    """Reject incomplete/non-comparable protocols before interpreting deltas."""
    versions: dict[str, dict[tuple, dict[str, Any]]] = {}
    sources = []
    checkpoint_contracts: dict[str, dict[str, Any]] = {}
    reference: dict[str, Any] | None = None
    dataset_paths: dict[str, str] = {}
    for version in ("old_data", "new_data"):
        rows: dict[tuple, dict[str, Any]] = {}
        for family in ("imputeformer", "bilstm"):
            path = root / version / f"{family}_benchmark.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            protocol = payload["protocol"]
            contract = {
                key: protocol.get(key)
                for key in (
                    "split",
                    "seeds",
                    "seed_mode",
                    "max_batches",
                    "cases",
                    "prediction_postprocessing",
                )
            }
            if (
                payload.get("schema_version") != 2
                or contract["split"] != "val"
                or not contract["prediction_postprocessing"]
            ):
                raise ValueError(
                    "Expected versioned validation artifacts, never test scores"
                )
            if reference is None:
                reference = contract
            if contract != reference:
                raise ValueError("Different validation protocols or postprocessing")
            checkpoints = protocol["checkpoints"]
            if (
                family in checkpoint_contracts
                and checkpoint_contracts[family] != checkpoints
            ):
                raise ValueError(
                    "Weights/training metadata differ across dataset versions"
                )
            checkpoint_contracts[family] = checkpoints
            if (
                version in dataset_paths
                and dataset_paths[version] != protocol["data_dir"]
            ):
                raise ValueError("Model families used different data roots")
            dataset_paths[version] = protocol["data_dir"]
            actual_variables = {row["variable"] for row in payload["results"]}
            if actual_variables != set(variables):
                raise ValueError(
                    f"Incomplete variable coverage in {path}: {actual_variables}"
                )
            for row in payload["results"]:
                if (
                    not all(math.isfinite(float(row[key])) for key in ("mae", "rmse"))
                    or row["n_hidden"] <= 0
                ):
                    raise ValueError("Invalid validation errors or hidden counts")
                if row["split"] != "val":
                    raise ValueError("Non-validation result row")
                model = f"{row['model']}/{row['model_variant']}"
                # Linear and circular interpolation are one family for pairing.
                if row["model"] == "interpolation":
                    model = "interpolation"
                key = (
                    row["variable"],
                    row["seed"],
                    row["mask_type"],
                    row["mask_parameter"],
                    model,
                )
                if key in rows:
                    if model != "interpolation" or any(
                        not math.isclose(
                            float(row[field]),
                            float(rows[key][field]),
                            rel_tol=0,
                            abs_tol=1e-6,
                        )
                        for field in ("n_hidden", "mae", "rmse")
                    ):
                        raise ValueError(
                            "Duplicate model row or unpaired interpolation baseline"
                        )
                else:
                    rows[key] = row
            sources.append({"path": str(path), "sha256": file_sha256(path)})
        if reference is None:
            raise ValueError("Missing validation protocol")
        expected = {
            (variable, seed, case["mask_type"], case["parameter"], model)
            for variable in variables
            for seed in reference["seeds"]
            for case in reference["cases"]
            for model in (*MODELS, "interpolation")
        }
        if set(rows) != expected:
            raise ValueError("Missing or extra model/seed/case rows")
        for key, row in rows.items():
            if row["n_hidden"] != rows[(*key[:4], "interpolation")]["n_hidden"]:
                raise ValueError("Model families evaluated different hidden counts")
        versions[version] = rows
    if dataset_paths["old_data"] == dataset_paths["new_data"]:
        raise ValueError("Expected distinct data roots")
    if reference is None:
        raise ValueError("Missing validation protocol")
    changes = []
    for key, new in versions["new_data"].items():
        old = versions["old_data"][key]
        changes.append(
            {
                "variable": key[0],
                "seed": key[1],
                "mask_type": key[2],
                "mask_parameter": key[3],
                "model": key[4],
                "old_mae": old["mae"],
                "new_mae": new["mae"],
                "mae_change": new["mae"] - old["mae"],
                "old_n_hidden": old["n_hidden"],
                "new_n_hidden": new["n_hidden"],
            }
        )
    checkpoint_hashes = {
        info["path"]: file_sha256(Path(info["path"]))
        for checkpoints in checkpoint_contracts.values()
        for info in checkpoints.values()
    }
    return {
        "schema_version": 1,
        "protocol": reference,
        "data_roots": dataset_paths,
        "sources": sources,
        "checkpoint_sha256": checkpoint_hashes,
        "limitation": "Exploratory validation of weights trained on old data.",
        "results": changes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists, choose a new report path")
    payload = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(payload["results"]),
                "checkpoints": len(payload["checkpoint_sha256"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
