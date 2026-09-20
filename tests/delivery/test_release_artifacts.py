"""Notebook inputs must identify one completed, comparable experiment."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from giano.prediction import prediction_policy
from giano.provenance import file_sha256

spec = importlib.util.spec_from_file_location(
    "release_artifacts",
    Path(__file__).resolve().parents[2] / "notebooks/release_artifacts.py",
)
assert spec is not None and spec.loader is not None
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


def _write(path, payload, identity=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    path.with_name(path.name + ".complete.json").write_text(
        json.dumps(
            {
                "output_sha256": file_sha256(path),
                "request": {"inputs": identity or {"data": "same"}},
            }
        )
    )


@pytest.mark.parametrize("problem", [None, "changed", "unfinished", "contract"])
def test_reader_requires_current_completed_output(tmp_path, problem):
    path = tmp_path / "result.json"
    _write(path, {"protocol": {"prediction_postprocessing": prediction_policy()}})
    if problem == "changed":
        path.write_text(path.read_text() + "\n")
    elif problem == "unfinished":
        path.with_name(path.name + ".complete.json").unlink()
    elif problem == "contract":
        _write(path, {"protocol": {"prediction_postprocessing": None}})
    if problem:
        with pytest.raises((ValueError, FileNotFoundError)):
            artifacts.read_completed(path)
    else:
        assert artifacts.read_completed(path)[1] == {"data": "same"}


@pytest.mark.parametrize(
    "problem", [None, "count", "reference", "snapshot", "test", "missing"]
)
def test_benchmark_reader_pairs_every_case_without_historical_fallback(
    tmp_path, monkeypatch, problem
):
    monkeypatch.setattr(artifacts, "VARIABLE_TYPE_NAMES", ("temperature",))
    monkeypatch.setattr(artifacts, "SEEDS", (42,))
    monkeypatch.setattr(
        artifacts,
        "load_benchmark_protocol",
        lambda _: SimpleNamespace(
            cases=[SimpleNamespace(mask_type="block", parameter=12)]
        ),
    )
    root = (
        tmp_path
        / artifacts.RELEASE_PATH
        / "validation/temperature/val_paired-seeds-42-43-44-45-46"
    )
    for family in ("imputeformer", "bilstm"):
        rows = [
            {
                "variable": "temperature",
                "split": "test" if problem == "test" else "val",
                "mask_type": "block",
                "mask_parameter": 12,
                "seed": 42,
                "model": model,
                "model_variant": variant,
                "mae": 1.0,
                "rmse": 2.0,
                "n_hidden": 4,
            }
            for model, variant in (
                (("giano", "imputeformer"), ("interpolation", "linear"))
                if family == "imputeformer"
                else (
                    ("bilstm", "fair"),
                    ("bilstm", "legacy_retrained"),
                    ("interpolation", "linear"),
                )
            )
        ]
        if family == "bilstm":
            if problem == "count":
                rows[0]["n_hidden"] = 5
            if problem == "reference":
                rows[-1]["mae"] = 3.0
            if problem == "missing":
                rows.pop(0)
        _write(
            root / f"{family}_benchmark.json",
            {
                "schema_version": 2,
                "protocol": {
                    "split": "val",
                    "seeds": [42],
                    "seed_mode": "paired_training_and_mask",
                    "prediction_postprocessing": prediction_policy(),
                },
                "results": rows,
            },
            {"data": family} if problem == "snapshot" else None,
        )
    if problem:
        with pytest.raises(ValueError):
            artifacts.load_benchmarks(tmp_path)
    else:
        rows, sources = artifacts.load_benchmarks(tmp_path)
        assert len(rows) == 4 and len(sources) == 2


def test_downstream_reader_has_no_development_fallback(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not complete"):
        artifacts.load_downstream(tmp_path, "temperature")
    path = tmp_path / artifacts.DOWNSTREAM_PATH / "downstream/temperature.json"
    _write(
        path,
        {
            "schema_version": 2,
            "protocol": {"repair_prediction_postprocessing": prediction_policy()},
            "records": [{"variable": "temperature"}],
        },
    )
    assert artifacts.load_downstream(tmp_path, "temperature")[1] == path
