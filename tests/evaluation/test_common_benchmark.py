from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from giano.evaluation.gapfill.cases import BenchmarkCase, load_benchmark_protocol
from giano.evaluation.gapfill.masks import (
    empirical_block_lengths,
    natural_gap_lengths,
    synthetic_mask,
)
from giano.evaluation.gapfill.results import (
    EvaluationResult,
    benchmark_artifact_paths,
    comparison_payload,
    markdown_table,
    rows_from_comparison,
    write_benchmark_artifact,
)


def test_benchmark_default_paths_are_scoped_by_variable_split_and_seeds() -> None:
    temperature, _ = benchmark_artifact_paths(
        "bilstm", ("temperature",), split="test", seeds=(42,)
    )
    wind, wind_markdown = benchmark_artifact_paths(
        "bilstm", ("wind_speed",), split="test", seeds=(44, 42, 43)
    )

    assert temperature == Path(
        "artifacts/evaluation/gapfill/temperature/test_seeds-42/bilstm_benchmark.json"
    )
    assert wind == Path(
        "artifacts/evaluation/gapfill/wind_speed/"
        "test_seeds-42-43-44/bilstm_benchmark.json"
    )
    assert wind_markdown == wind.with_suffix(".md")

    paired, _ = benchmark_artifact_paths(
        "imputeformer",
        ("temperature",),
        split="test",
        seeds=(42, 43, 44),
        seed_mode="paired_training_and_mask",
    )
    assert paired == Path(
        "artifacts/evaluation/gapfill/temperature/"
        "test_paired-seeds-42-43-44/imputeformer_benchmark.json"
    )


def test_shared_masks_are_deterministic_and_preserve_context() -> None:
    observed = np.ones((24, 4), dtype=bool)
    first = synthetic_mask(observed, np.random.default_rng(17), "point", 0.5)
    second = synthetic_mask(observed, np.random.default_rng(17), "point", 0.5)

    assert np.array_equal(first, second)
    assert first.any()
    assert np.all((observed & ~first).sum(axis=0) >= 2)


def test_terminal_mask_never_hides_values_after_the_window_origin() -> None:
    observed = np.ones((24, 5), dtype=bool)
    mask = synthetic_mask(observed, np.random.default_rng(5), "terminal", 6)

    assert not mask[:-6].any()
    assert mask[-6:].any()


def test_empirical_mask_uses_only_supplied_train_gap_distribution() -> None:
    values = np.ones((20, 2), dtype=np.float64)
    values[2:4, 0] = np.nan
    values[8:13, 1] = np.nan
    values[16:, 0] = np.nan
    bounds = np.array([[0, 19], [0, 19]])

    assert natural_gap_lengths(values, bounds, 15).tolist() == [2, 5]
    lengths = empirical_block_lengths(
        values,
        bounds,
        15,
        quantile_cap=1.0,
        max_length=10,
        samples=8,
    )
    mask = synthetic_mask(
        np.ones((12, 1), dtype=bool),
        np.random.default_rng(3),
        "empirical",
        1.0,
        empirical_lengths=lengths,
    )

    assert 2 <= int(mask.sum()) <= 5


def test_project_protocol_freezes_unique_cases_and_terminal_gaps() -> None:
    protocol = load_benchmark_protocol(
        Path(__file__).resolve().parents[2] / "config.yaml"
    )

    assert protocol.split == "test"
    assert len(protocol.cases) == len(set(protocol.cases))
    assert BenchmarkCase("terminal", 24) in protocol.cases


def test_common_rows_write_strict_json_and_generated_markdown(tmp_path: Path) -> None:
    case = BenchmarkCase("block", 6)
    result = EvaluationResult(
        model_mae=1.0,
        baseline_mae=2.0,
        model_rmse=1.5,
        baseline_rmse=2.5,
        variation_ratio=float("nan"),
        count=12,
    )
    comparison = comparison_payload(
        result,
        variable="temperature",
        split="test",
        seed=42,
        case=case,
        runtime_ms=3.5,
    )
    rows = rows_from_comparison(
        comparison,
        model="giano",
        model_variant="imputeformer",
        variable="temperature",
        seed=42,
        split="test",
        case=case,
        runtime_ms=3.5,
        parameter_count=100,
    )
    output = tmp_path / "benchmark.json"
    payload = write_benchmark_artifact(
        output,
        rows,
        protocol={"seeds": [42]},
        comparisons=[comparison],
    )

    reparsed = json.loads(output.read_text(encoding="utf-8"))
    assert payload == reparsed
    assert reparsed["schema_version"] == 2
    assert reparsed["comparisons"][0]["variation_ratio"] is None
    assert "giano/imputeformer" in markdown_table(rows)


def test_checkpoint_history_with_undefined_circular_variation_is_strict_json(
    tmp_path: Path,
) -> None:
    validation = {"model_mae": 39.5, "variation_ratio": float("nan")}
    protocol = {
        "checkpoints": {
            "seed-42/wind_direction": {
                "training_run": {"history": [{"validation": validation}]}
            }
        }
    }
    output = tmp_path / "circular.json"
    payload = write_benchmark_artifact(output, [], protocol=protocol)
    history = payload["protocol"]["checkpoints"]["seed-42/wind_direction"][
        "training_run"
    ]["history"]
    assert history[0]["validation"] == {"model_mae": 39.5, "variation_ratio": None}
    assert json.loads(output.read_text()) == payload
    assert np.isnan(validation["variation_ratio"])
    validation["model_mae"] = float("nan")
    with pytest.raises(ValueError, match="Out of range float"):
        write_benchmark_artifact(tmp_path / "invalid.json", [], protocol=protocol)
