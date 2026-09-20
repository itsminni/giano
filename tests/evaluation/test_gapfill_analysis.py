from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from giano.evaluation.gapfill.analyze import (
    _load_paired_rows,
    aggregate_scores,
    natural_gap_statistics,
    paired_seed_differences,
)
from giano.evaluation.gapfill.cases import BenchmarkCase
from giano.evaluation.gapfill.detailed import pair_units, paired_unit_interval
from giano.evaluation.gapfill.results import EvaluationUnit
from giano.prediction import prediction_policy
from giano.spatiotemporal_dataset import SpatiotemporalPanel


def _row(model: str, variant: str, seed: int, mae: float) -> dict[str, object]:
    return {
        "model": model,
        "model_variant": variant,
        "variable": "temperature",
        "seed": seed,
        "mask_type": "block",
        "mask_parameter": 6,
        "mae": mae,
        "rmse": mae * 2,
        "runtime_ms": 3.0,
        "parameter_count": 10,
    }


def test_aggregate_and_paired_seed_bootstrap_preserve_pairing() -> None:
    rows = [
        _row("giano", "imputeformer", 42, 2.0),
        _row("giano", "imputeformer", 43, 3.0),
        _row("bilstm", "fair", 42, 1.0),
        _row("bilstm", "fair", 43, 2.0),
    ]

    aggregates = aggregate_scores(rows)
    giano = next(item for item in aggregates if item.model == "giano/imputeformer")
    assert giano.seeds == (42, 43)
    assert giano.mean_mae == 2.5
    differences = paired_seed_differences(
        rows,
        reference_model="giano/imputeformer",
        candidate_model="bilstm/fair",
        bootstrap_samples=100,
        seed=7,
    )
    assert len(differences) == 1
    assert differences[0].mean_mae_difference == -1.0
    assert differences[0].candidate_wins == 2


def test_natural_gap_statistics_use_only_train_and_station_coverage() -> None:
    values = np.ones((20, 2), dtype=np.float32)
    values[2:4, 0] = np.nan
    values[10:13, 0] = np.nan
    values[5, 1] = np.nan
    times = np.datetime64("2020-01-01T00", "ns") + np.arange(20) * np.timedelta64(
        1, "h"
    )
    panel = SpatiotemporalPanel(
        variable="temperature",
        times=times,
        station_ids=("A", "B"),
        coordinates=np.zeros((2, 2), dtype=np.float32),
        values=values,
        auxiliary=None,
        coverage_bounds=np.asarray(((0, 19), (3, 19))),
        source_paths=(Path("A.nc"), Path("B.nc")),
    )

    result = natural_gap_statistics(panel, train_ratio=0.5)

    assert result["gap_count"] == 2
    assert result["missing_hours"] == 3
    assert result["max_gap_hours"] == 2
    assert result["stations"][1]["gap_count"] == 1


def test_load_paired_rows_supports_combined_variable_artifacts(
    tmp_path: Path,
) -> None:
    variables = ("temperature", "humidity")
    run_root = tmp_path / "humidity_temperature" / "test_paired-seeds-42-43"
    run_root.mkdir(parents=True)
    protocol = {
        "seed_mode": "paired_training_and_mask",
        "seeds": [42, 43],
    }
    giano_rows = [
        {**_row("giano", "imputeformer", seed, 1.0), "variable": variable}
        for variable in variables
        for seed in (42, 43)
    ]
    bilstm_rows = [
        {**_row("bilstm", "fair", seed, 0.9), "variable": variable}
        for variable in variables
        for seed in (42, 43)
    ]
    interpolation_rows = [
        {**_row("interpolation", "linear", seed, 1.1), "variable": variable}
        for variable in variables
        for seed in (42, 43)
    ]
    comparison = {
        "model_variant": "fair",
        "variable": "temperature",
        "variation_ratio": 0.5,
    }
    (run_root / "imputeformer_benchmark.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "protocol": protocol,
                "results": giano_rows + interpolation_rows,
                "comparisons": [],
            }
        ),
        encoding="utf-8",
    )
    (run_root / "bilstm_benchmark.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "protocol": protocol,
                "results": bilstm_rows + interpolation_rows,
                "comparisons": [comparison],
            }
        ),
        encoding="utf-8",
    )

    rows, sources, variation = _load_paired_rows(tmp_path, variables, (42, 43), "test")

    assert len(rows) == 12
    assert len(sources) == 2
    assert variation == {"temperature:bilstm/fair": 0.5}


def test_load_paired_rows_rejects_mixed_postprocessing(tmp_path: Path) -> None:
    root = tmp_path / "temperature" / "test_paired-seeds-42"
    root.mkdir(parents=True)
    protocol: dict[str, object] = {
        "seed_mode": "paired_training_and_mask",
        "seeds": [42],
    }
    payload = {"schema_version": 2, "protocol": protocol, "results": []}
    (root / "imputeformer_benchmark.json").write_text(json.dumps(payload))
    protocol["prediction_postprocessing"] = prediction_policy()
    (root / "bilstm_benchmark.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different prediction postprocessing"):
        _load_paired_rows(tmp_path, ("temperature",), (42,), "test")
    (root / "imputeformer_benchmark.json").write_text(json.dumps(payload))
    rows, sources, _ = _load_paired_rows(tmp_path, ("temperature",), (42,), "test")
    assert not rows
    assert len(sources) == 2


def _unit(start: int, station: str, model_error: float) -> EvaluationUnit:
    return EvaluationUnit(
        start_index=start,
        station=station,
        n_hidden=2,
        model_abs_sum=model_error,
        baseline_abs_sum=6.0,
        model_squared_sum=model_error**2,
        baseline_squared_sum=18.0,
    )


def test_detailed_units_remain_paired_for_hierarchical_bootstrap() -> None:
    rows = []
    for seed in (42, 43, 44):
        giano = [_unit(0, "A", 2.0), _unit(72, "B", 4.0)]
        bilstm = [_unit(0, "A", 4.0), _unit(72, "B", 6.0)]
        rows.extend(
            pair_units(
                giano,
                bilstm,
                variable="temperature",
                seed=seed,
                case=BenchmarkCase("block", 12),
            )
        )

    observed, low, high = paired_unit_interval(
        rows,
        candidate="bilstm_fair",
        reference="giano",
        samples=100,
        seed=3,
    )

    assert observed == 1.0
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)
