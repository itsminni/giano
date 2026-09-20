"""Rolling-origin utility evaluation that never exposes future values."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from giano.downstream.ridge_forecaster import (
    AutoregressiveRidgeForecaster,
    RidgeForecastConfig,
    complete_origins,
)
from giano.evaluation.gapfill.cases import BenchmarkCase
from giano.evaluation.gapfill.masks import synthetic_mask

HistoryRepairer = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ForecastRecord:
    """One method prediction at one origin and horizon."""

    method: str
    origin_index: int
    origin_time: str
    horizon: int
    target: float
    prediction: float
    absolute_error: float
    squared_error: float
    variable: str = ""
    station: str = ""
    seed: int = 0
    mask_type: str = ""
    mask_parameter: int | float = 0
    mask_parameter_unit: str = ""


@dataclass(frozen=True)
class ForecastSummary:
    """Aggregate downstream utility for one method and horizon."""

    method: str
    horizon: int
    mae: float
    rmse: float
    n_origins: int
    recovery_ratio: float | None
    mae_ci_low: float
    mae_ci_high: float
    variable: str = ""
    station: str = ""
    seed: int = 0
    mask_type: str = ""
    mask_parameter: int | float = 0
    mask_parameter_unit: str = ""
    mae_difference_vs_corrupted: float | None = None
    difference_ci_low: float | None = None
    difference_ci_high: float | None = None


@dataclass(frozen=True)
class ForecastComparison:
    """Paired candidate-minus-reference forecast MAE on identical origins."""

    variable: str
    station: str
    mask_type: str
    mask_parameter: int | float
    horizon: int
    candidate: str
    reference: str
    seeds: tuple[int, ...]
    n_pairs: int
    mean_absolute_error_difference: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class RollingOriginConfig:
    """Frozen test-origin and bootstrap settings."""

    stride: int = 24
    max_origins: int = 200
    bootstrap_samples: int = 1000
    seed: int = 42

    def validate(self) -> None:
        """Validate evaluation limits and statistical repetition counts."""
        if min(self.stride, self.max_origins, self.bootstrap_samples) <= 0:
            raise ValueError("rolling-origin limits must be positive")


def load_forecasting_config(
    path: Path,
) -> tuple[RidgeForecastConfig, RollingOriginConfig]:
    """Load the complete downstream recipe from the application YAML."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration top-level must be a mapping")
    section = raw.get("downstream", {})
    if not isinstance(section, dict):
        raise ValueError("downstream configuration must be a mapping")
    ridge_raw = section.get("ridge", {})
    rolling_raw = section.get("rolling_origin", {})
    if not isinstance(ridge_raw, dict) or not isinstance(rolling_raw, dict):
        raise ValueError("downstream ridge/rolling_origin must be mappings")
    ridge_values = {
        key: value
        for key, value in ridge_raw.items()
        if key in RidgeForecastConfig.__dataclass_fields__
    }
    if "horizons" in ridge_values:
        ridge_values["horizons"] = tuple(ridge_values["horizons"])
    rolling_values = {
        key: value
        for key, value in rolling_raw.items()
        if key in RollingOriginConfig.__dataclass_fields__
    }
    ridge = RidgeForecastConfig(**ridge_values)
    rolling = RollingOriginConfig(**rolling_values)
    ridge.validate()
    rolling.validate()
    return ridge, rolling


def _linear_interpolation(history: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    positions = np.arange(len(history), dtype=np.float64)
    visible = ~hidden
    if visible.sum() < 2:
        raise ValueError("interpolation requires at least two visible history values")
    repaired = history.copy()
    repaired[hidden] = np.interp(
        positions[hidden], positions[visible], history[visible]
    )
    return repaired


def _circular_interpolation(history: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    positions = np.arange(len(history), dtype=np.float64)
    visible = ~hidden
    if visible.sum() < 2:
        raise ValueError("circular interpolation requires two visible history values")
    unwrapped = np.unwrap(np.deg2rad(history[visible]))
    repaired = history.copy()
    repaired[hidden] = np.mod(
        np.rad2deg(np.interp(positions[hidden], positions[visible], unwrapped)),
        360.0,
    )
    return repaired


def prepare_origin_histories(
    values: np.ndarray,
    times: np.ndarray,
    origin: int,
    history_length: int,
    case: BenchmarkCase,
    *,
    seed: int,
    direction: bool = False,
    repairers: Mapping[str, HistoryRepairer] | None = None,
    empirical_lengths: tuple[int, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Create same-mask histories using data no later than ``origin``."""
    series = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(times).astype("datetime64[ns]")
    start = origin - history_length + 1
    if start < 0 or origin >= len(series) or len(series) != len(timestamps):
        raise ValueError("origin does not admit the requested aligned history")
    clean = series[start : origin + 1].copy()
    history_times = timestamps[start : origin + 1].copy()
    if not np.isfinite(clean).all():
        raise ValueError("rolling-origin history must be fully observed before masking")
    mask = synthetic_mask(
        np.ones((history_length, 1), dtype=bool),
        np.random.default_rng(seed + origin),
        case.mask_type,
        case.parameter,
        empirical_lengths=empirical_lengths,
    )[:, 0]
    if not mask.any():
        raise RuntimeError("rolling-origin case produced no hidden history values")
    visible = ~mask
    neutral = float(np.median(clean[visible]))
    corrupted = clean.copy()
    corrupted[mask] = neutral
    interpolation = (
        _circular_interpolation(clean, mask)
        if direction
        else _linear_interpolation(clean, mask)
    )
    histories = {
        "clean": clean,
        "corrupted": corrupted,
        "interpolation": interpolation,
    }
    for name, repairer in (repairers or {}).items():
        if name in histories or not name:
            raise ValueError(f"duplicate or empty forecasting method name: {name}")
        repaired = np.asarray(
            repairer(corrupted.copy(), visible.copy(), history_times.copy()),
            dtype=np.float64,
        )
        if repaired.shape != clean.shape or not np.isfinite(repaired).all():
            raise ValueError(f"repairer {name} returned invalid history")
        if not np.allclose(repaired[visible], clean[visible], rtol=0.0, atol=1e-8):
            raise ValueError(f"repairer {name} changed visible observations")
        histories[name] = repaired
    return histories


def evaluate_rolling_origin(  # noqa: PLR0913
    values: np.ndarray,
    times: np.ndarray,
    *,
    train_end: int,
    test_start: int,
    forecaster: AutoregressiveRidgeForecaster,
    case: BenchmarkCase,
    config: RollingOriginConfig | None = None,
    direction: bool = False,
    repairers: Mapping[str, HistoryRepairer] | None = None,
    empirical_lengths: tuple[int, ...] | None = None,
    variable: str = "",
    station: str = "",
    experiment_seed: int | None = None,
) -> list[ForecastRecord]:
    """Fit once on train, freeze, then score identical rolling origins."""
    settings = config or RollingOriginConfig()
    settings.validate()
    series = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(times).astype("datetime64[ns]")
    forecaster.fit_series(series, timestamps, train_end)
    origins = complete_origins(
        series,
        history_length=forecaster.config.history_length,
        horizons=forecaster.config.horizons,
        start=test_start,
        end=len(series),
        stride=settings.stride,
    )[: settings.max_origins]
    records: list[ForecastRecord] = []
    horizon_offsets = np.asarray(forecaster.config.horizons)
    for origin in origins:
        targets = series[origin + horizon_offsets]
        histories = prepare_origin_histories(
            series,
            timestamps,
            origin,
            forecaster.config.history_length,
            case,
            seed=settings.seed,
            direction=direction,
            repairers=repairers,
            empirical_lengths=empirical_lengths,
        )
        predictions = {
            method: forecaster.predict_history(history, timestamps[origin])
            for method, history in histories.items()
        }
        for method, method_predictions in predictions.items():
            for horizon, target, prediction in zip(
                forecaster.config.horizons,
                targets,
                method_predictions,
                strict=True,
            ):
                error = float(prediction - target)
                if direction:
                    error = float((error + 180.0) % 360.0 - 180.0)
                records.append(
                    ForecastRecord(
                        method=method,
                        origin_index=int(origin),
                        origin_time=np.datetime_as_string(timestamps[origin], unit="h"),
                        horizon=horizon,
                        target=float(target),
                        prediction=float(prediction),
                        absolute_error=abs(error),
                        squared_error=error * error,
                        variable=variable,
                        station=station,
                        seed=(
                            settings.seed
                            if experiment_seed is None
                            else experiment_seed
                        ),
                        mask_type=case.mask_type,
                        mask_parameter=case.parameter,
                        mask_parameter_unit=case.parameter_unit,
                    )
                )
    if not records:
        raise RuntimeError("rolling-origin evaluation found no complete test origins")
    return records


def _bootstrap_mean_interval(
    errors: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(errors), size=(samples, len(errors)))
    means = errors[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def summarize_forecasts(
    records: list[ForecastRecord],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> list[ForecastSummary]:
    """Aggregate per-horizon errors, intervals, and recovery ratios."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    grouped: dict[
        tuple[str, str, int, str, int | float, str, int],
        list[ForecastRecord],
    ] = {}
    for record in records:
        grouped.setdefault(
            (
                record.variable,
                record.station,
                record.seed,
                record.mask_type,
                record.mask_parameter,
                record.method,
                record.horizon,
            ),
            [],
        ).append(record)
    summaries: list[ForecastSummary] = []
    for group_index, (key, group) in enumerate(sorted(grouped.items())):
        (
            variable,
            station,
            experiment_seed,
            mask_type,
            mask_parameter,
            method,
            horizon,
        ) = key
        scope = (variable, station, experiment_seed, mask_type, mask_parameter)
        clean_key = (*scope, "clean", horizon)
        corrupted_key = (*scope, "corrupted", horizon)
        if clean_key not in grouped or corrupted_key not in grouped:
            raise ValueError("forecast records require clean and corrupted methods")
        errors = np.asarray([record.absolute_error for record in group])
        squared = np.asarray([record.squared_error for record in group])
        clean_group = grouped[clean_key]
        corrupted_group = grouped[corrupted_key]
        clean_mae = np.mean([record.absolute_error for record in clean_group])
        corrupted_mae = np.mean([record.absolute_error for record in corrupted_group])
        denominator = corrupted_mae - clean_mae
        recovery = (
            float((corrupted_mae - errors.mean()) / denominator)
            if abs(denominator) > 1e-12
            else None
        )
        ci_low, ci_high = _bootstrap_mean_interval(
            errors,
            samples=bootstrap_samples,
            seed=seed + group_index,
        )
        by_origin = {record.origin_index: record.absolute_error for record in group}
        corrupted_by_origin = {
            record.origin_index: record.absolute_error for record in corrupted_group
        }
        if by_origin.keys() != corrupted_by_origin.keys():
            raise ValueError("paired downstream methods must use identical origins")
        differences = np.asarray(
            [
                by_origin[origin] - corrupted_by_origin[origin]
                for origin in sorted(by_origin)
            ],
            dtype=np.float64,
        )
        difference_low, difference_high = _bootstrap_mean_interval(
            differences,
            samples=bootstrap_samples,
            seed=seed + len(grouped) + group_index,
        )
        summaries.append(
            ForecastSummary(
                method=method,
                horizon=horizon,
                mae=float(errors.mean()),
                rmse=float(math.sqrt(squared.mean())),
                n_origins=len(group),
                recovery_ratio=recovery,
                mae_ci_low=ci_low,
                mae_ci_high=ci_high,
                variable=variable,
                station=station,
                seed=experiment_seed,
                mask_type=mask_type,
                mask_parameter=mask_parameter,
                mask_parameter_unit=(group[0].mask_parameter_unit if group else ""),
                mae_difference_vs_corrupted=float(differences.mean()),
                difference_ci_low=difference_low,
                difference_ci_high=difference_high,
            )
        )
    return summaries


def paired_method_comparisons(
    records: list[ForecastRecord],
    *,
    reference: str,
    candidates: tuple[str, ...],
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> list[ForecastComparison]:
    """Compare methods using hierarchical paired seed/origin resampling."""
    if bootstrap_samples <= 0 or not candidates:
        raise ValueError("paired comparisons require candidates and samples")
    indexed: dict[
        tuple[str, str, str, int | float, int, str, int, int],
        float,
    ] = {}
    for record in records:
        key = (
            record.variable,
            record.station,
            record.mask_type,
            record.mask_parameter,
            record.horizon,
            record.method,
            record.seed,
            record.origin_index,
        )
        if key in indexed:
            raise ValueError("duplicate downstream method/origin record")
        indexed[key] = record.absolute_error
    scopes = sorted({key[:5] for key in indexed})
    comparisons = []
    for scope_index, scope in enumerate(scopes):
        for candidate_index, candidate in enumerate(candidates):
            by_seed: dict[int, np.ndarray] = {}
            available_seeds = sorted(
                {key[6] for key in indexed if key[:5] == scope and key[5] == reference}
            )
            for item_seed in available_seeds:
                reference_origins = {
                    key[7]
                    for key in indexed
                    if key[:7] == (*scope, reference, item_seed)
                }
                candidate_origins = {
                    key[7]
                    for key in indexed
                    if key[:7] == (*scope, candidate, item_seed)
                }
                if reference_origins != candidate_origins or not reference_origins:
                    raise ValueError("paired methods do not share identical origins")
                by_seed[item_seed] = np.asarray(
                    [
                        indexed[(*scope, candidate, item_seed, origin)]
                        - indexed[(*scope, reference, item_seed, origin)]
                        for origin in sorted(reference_origins)
                    ],
                    dtype=np.float64,
                )
            if not by_seed:
                raise ValueError(f"reference method is absent: {reference}")
            observed = float(np.concatenate(list(by_seed.values())).mean())
            rng = np.random.default_rng(
                seed + scope_index * len(candidates) + candidate_index
            )
            bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
            seed_values = tuple(by_seed)
            for sample in range(bootstrap_samples):
                selected_seeds = rng.choice(
                    seed_values,
                    len(seed_values),
                    replace=True,
                )
                total = 0.0
                count = 0
                for selected_seed in selected_seeds:
                    differences = by_seed[int(selected_seed)]
                    indices = rng.integers(0, len(differences), size=len(differences))
                    total += float(differences[indices].sum())
                    count += len(differences)
                bootstrap[sample] = total / count
            low, high = np.quantile(bootstrap, (0.025, 0.975))
            variable, station, mask_type, mask_parameter, horizon = scope
            comparisons.append(
                ForecastComparison(
                    variable=variable,
                    station=station,
                    mask_type=mask_type,
                    mask_parameter=mask_parameter,
                    horizon=horizon,
                    candidate=candidate,
                    reference=reference,
                    seeds=tuple(available_seeds),
                    n_pairs=sum(len(item) for item in by_seed.values()),
                    mean_absolute_error_difference=observed,
                    ci_low=float(low),
                    ci_high=float(high),
                )
            )
    return comparisons


def write_forecasting_artifact(
    path: Path,
    records: list[ForecastRecord],
    summaries: list[ForecastSummary],
    *,
    protocol: dict[str, object],
    comparisons: list[ForecastComparison] | None = None,
) -> dict[str, object]:
    """Write downstream records and summaries below the forecasting folder."""
    payload: dict[str, object] = {
        "schema_version": 2,
        "protocol": protocol,
        "records": [asdict(record) for record in records],
        "summaries": [asdict(summary) for summary in summaries],
    }
    if comparisons is not None:
        payload["paired_method_comparisons"] = [
            asdict(comparison) for comparison in comparisons
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return payload
