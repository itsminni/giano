from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from giano.downstream.ridge_forecaster import (
    AutoregressiveRidgeForecaster,
    RidgeForecastConfig,
    RidgeRegressor,
)
from giano.evaluation.forecasting.protocol import (
    ForecastRecord,
    RollingOriginConfig,
    evaluate_rolling_origin,
    load_forecasting_config,
    paired_method_comparisons,
    prepare_origin_histories,
    summarize_forecasts,
    write_forecasting_artifact,
)
from giano.evaluation.gapfill.cases import BenchmarkCase
from giano.evaluation.gapfill.masks import empirical_block_lengths


def test_empirical_downstream_cap_preserves_two_visible_values():
    values = np.ones((300, 1))
    values[10:150] = np.nan
    lengths = empirical_block_lengths(
        values,
        np.array([[0, 299]]),
        200,
        quantile_cap=0.95,
        max_length=71,
    )
    assert max(lengths) == 70
    clean, times = _series()
    histories = prepare_origin_histories(
        clean,
        times,
        300,
        72,
        BenchmarkCase("empirical", 0.95),
        seed=42,
        empirical_lengths=lengths,
    )
    assert all(np.isfinite(history).all() for history in histories.values())


def test_origin_cap_applies_after_filtering_incomplete_histories(tmp_path):
    values, times = _series()
    values[300:330] = np.nan
    records = evaluate_rolling_origin(
        values,
        times,
        train_end=280,
        test_start=300,
        forecaster=AutoregressiveRidgeForecaster(
            RidgeForecastConfig(history_length=12, horizons=(1, 6))
        ),
        case=BenchmarkCase("block", 3),
        config=RollingOriginConfig(stride=1, max_origins=5, bootstrap_samples=10),
    )
    assert len({row.origin_index for row in records}) == 5
    assert min(row.origin_index for row in records) == 341
    write_forecasting_artifact(tmp_path / "result.json", records, [], protocol={})


def _series(length: int = 480) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(length, dtype=np.float64)
    values = 10.0 + positions * 0.02 + np.sin(positions * 2.0 * np.pi / 24.0)
    times = np.datetime64("2020-01-01T00", "ns") + positions.astype(
        int
    ) * np.timedelta64(1, "h")
    return values, times


def test_numpy_ridge_fits_multi_output_targets() -> None:
    design = np.arange(20, dtype=np.float64)[:, None]
    targets = np.column_stack((2.0 * design[:, 0] + 1.0, -design[:, 0] + 3.0))

    model = RidgeRegressor(alpha=1e-10).fit(design, targets)
    prediction = model.predict(np.array([[21.0]]))[0]

    assert np.allclose(prediction, [43.0, -18.0], atol=1e-7)


def test_downstream_recipe_is_loaded_from_the_project_config() -> None:
    ridge, rolling = load_forecasting_config(
        Path(__file__).resolve().parents[2] / "config.yaml"
    )

    assert ridge.history_length == 72
    assert ridge.horizons == (1, 6, 12, 24)
    assert rolling.seed == 42


def test_rolling_origin_uses_one_frozen_forecaster_for_every_method() -> None:
    values, times = _series()
    forecaster = AutoregressiveRidgeForecaster(
        RidgeForecastConfig(history_length=48, horizons=(1, 6, 12), alpha=0.1)
    )
    records = evaluate_rolling_origin(
        values,
        times,
        train_end=300,
        test_start=320,
        forecaster=forecaster,
        case=BenchmarkCase("terminal", 12),
        config=RollingOriginConfig(
            stride=12,
            max_origins=8,
            bootstrap_samples=50,
            seed=7,
        ),
    )

    counts: dict[tuple[str, int], int] = {}
    for record in records:
        counts[(record.method, record.horizon)] = (
            counts.get((record.method, record.horizon), 0) + 1
        )
    assert forecaster.training_samples > 0
    assert set(method for method, _horizon in counts) == {
        "clean",
        "corrupted",
        "interpolation",
    }
    assert len(set(counts.values())) == 1

    summaries = summarize_forecasts(records, bootstrap_samples=50, seed=7)
    assert len(summaries) == 9
    assert all(summary.n_origins == 8 for summary in summaries)
    assert all(
        summary.mae_ci_low <= summary.mae <= summary.mae_ci_high
        for summary in summaries
    )


def test_recovery_ratio_uses_clean_and_corrupted_references() -> None:
    records = []
    errors = {"clean": 1.0, "corrupted": 3.0, "method": 2.0}
    for method, error in errors.items():
        for origin in (10, 20):
            records.append(
                ForecastRecord(
                    method=method,
                    origin_index=origin,
                    origin_time="2020-01-01T00",
                    horizon=1,
                    target=0.0,
                    prediction=error,
                    absolute_error=error,
                    squared_error=error**2,
                )
            )

    summaries = summarize_forecasts(records, bootstrap_samples=20)
    method_summary = next(
        summary for summary in summaries if summary.method == "method"
    )

    assert method_summary.recovery_ratio == 0.5


def test_forecasting_artifact_is_strict_json(tmp_path: Path) -> None:
    records = [
        ForecastRecord("clean", 10, "2020-01-01T00", 1, 2.0, 2.0, 0.0, 0.0),
        ForecastRecord("corrupted", 10, "2020-01-01T00", 1, 2.0, 3.0, 1.0, 1.0),
    ]
    summaries = summarize_forecasts(records, bootstrap_samples=10)
    output = tmp_path / "forecasting" / "utility.json"

    payload = write_forecasting_artifact(
        output,
        records,
        summaries,
        protocol={"future_values_visible_to_imputer": False},
    )

    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_method_comparison_pairs_seeds_and_origins() -> None:
    records = []
    for seed in (42, 43, 44):
        for origin in (10, 20, 30):
            for method, error in (("giano", 1.0), ("bilstm_fair", 2.0)):
                records.append(
                    ForecastRecord(
                        method=method,
                        origin_index=origin,
                        origin_time="2020-01-01T00",
                        horizon=1,
                        target=0.0,
                        prediction=error,
                        absolute_error=error,
                        squared_error=error**2,
                        variable="temperature",
                        station="A",
                        seed=seed,
                        mask_type="block",
                        mask_parameter=12,
                    )
                )

    comparisons = paired_method_comparisons(
        records,
        reference="giano",
        candidates=("bilstm_fair",),
        bootstrap_samples=50,
    )

    assert comparisons[0].n_pairs == 9
    assert comparisons[0].mean_absolute_error_difference == 1.0
    assert comparisons[0].ci_low == comparisons[0].ci_high == 1.0


def test_future_changes_cannot_affect_histories_given_to_repairers() -> None:
    length = 160
    origin = 100
    history_length = 48
    values = np.sin(np.arange(length, dtype=np.float64) / 4.0)
    times = np.datetime64("2020-01-01T00", "ns") + np.arange(length) * np.timedelta64(
        1, "h"
    )
    changed_future = values.copy()
    changed_future[origin + 1 :] += 10_000.0
    seen_latest_times: list[np.datetime64] = []

    def repairer(
        corrupted: np.ndarray,
        visible: np.ndarray,
        history_times: np.ndarray,
    ) -> np.ndarray:
        seen_latest_times.append(history_times[-1])
        repaired = corrupted.copy()
        positions = np.arange(len(repaired))
        repaired[~visible] = np.interp(
            positions[~visible], positions[visible], repaired[visible]
        )
        return repaired

    case = BenchmarkCase("block", 12)
    original = prepare_origin_histories(
        values,
        times,
        origin,
        history_length,
        case,
        seed=13,
        repairers={"candidate": repairer},
    )
    perturbed = prepare_origin_histories(
        changed_future,
        times,
        origin,
        history_length,
        case,
        seed=13,
        repairers={"candidate": repairer},
    )

    assert original.keys() == perturbed.keys()
    for method in original:
        assert np.array_equal(original[method], perturbed[method])
    assert seen_latest_times
    assert all(latest == times[origin] for latest in seen_latest_times)
