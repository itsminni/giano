"""Small NumPy autoregressive Ridge forecaster for downstream evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def complete_origins(
    values: np.ndarray,
    *,
    history_length: int,
    horizons: tuple[int, ...],
    start: int,
    end: int,
    stride: int = 1,
) -> np.ndarray:
    """Return origins with finite histories/targets strictly before ``end``.

    Selection and Ridge fitting share this availability-only rule. Prefix sums
    avoid constructing a large station-by-window-by-history tensor.
    """
    origins = np.arange(max(start, history_length - 1), end - horizons[-1], stride)
    if not len(origins):
        return origins
    finite = np.isfinite(values)
    missing = np.concatenate(([0], np.cumsum(~finite)))
    complete_history = (
        missing[origins + 1] - missing[origins - history_length + 1]
    ) == 0
    complete_targets = finite[origins[:, None] + np.asarray(horizons)].all(axis=1)
    return origins[complete_history & complete_targets]


@dataclass(frozen=True)
class RidgeForecastConfig:
    """Frozen design choices for the downstream forecasting oracle."""

    history_length: int = 72
    horizons: tuple[int, ...] = (1, 6, 12, 24)
    alpha: float = 1.0
    include_calendar: bool = True

    def validate(self) -> None:
        """Reject ambiguous or numerically invalid forecasting recipes."""
        if self.history_length < 2:
            raise ValueError("history_length must be at least two")
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("forecast horizons must be positive")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("forecast horizons must be sorted and unique")
        if self.alpha < 0:
            raise ValueError("Ridge alpha cannot be negative")


class RidgeRegressor:
    """Multi-output Ridge with standardized inputs and an unpenalized intercept."""

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("Ridge alpha cannot be negative")
        self.alpha = float(alpha)
        self.x_mean: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None

    def fit(self, design: np.ndarray, targets: np.ndarray) -> RidgeRegressor:
        """Fit a deterministic closed-form Ridge system."""
        x = np.asarray(design, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError(
                "design and targets must be aligned two-dimensional arrays"
            )
        if len(x) < 2 or x.shape[1] < 1 or y.shape[1] < 1:
            raise ValueError("Ridge requires at least two rows and one feature/target")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Ridge training arrays must be finite")
        x_mean = x.mean(axis=0)
        raw_scale = x.std(axis=0)
        x_scale = np.where(raw_scale > 1e-12, raw_scale, 1.0)
        y_mean = y.mean(axis=0)
        self.x_mean = x_mean
        self.x_scale = x_scale
        self.y_mean = y_mean
        standardized = (x - x_mean) / x_scale
        centered_targets = y - y_mean
        gram = standardized.T @ standardized
        penalty = np.eye(gram.shape[0], dtype=np.float64) * self.alpha
        try:
            self.coefficients = np.linalg.solve(
                gram + penalty,
                standardized.T @ centered_targets,
            )
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.pinv(gram + penalty) @ (
                standardized.T @ centered_targets
            )
        return self

    def predict(self, design: np.ndarray) -> np.ndarray:
        """Predict all fitted outputs for one or more design rows."""
        if (
            self.x_mean is None
            or self.x_scale is None
            or self.y_mean is None
            or self.coefficients is None
        ):
            raise RuntimeError("RidgeRegressor must be fitted before prediction")
        x_mean = self.x_mean
        x_scale = self.x_scale
        y_mean = self.y_mean
        coefficients = self.coefficients
        x = np.asarray(design, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != len(x_mean):
            raise ValueError("prediction design has the wrong feature count")
        if not np.isfinite(x).all():
            raise ValueError("prediction design must be finite")
        standardized = (x - x_mean) / x_scale
        return standardized @ coefficients + y_mean


class AutoregressiveRidgeForecaster:
    """Frozen multi-horizon forecaster shared by every repaired history."""

    def __init__(self, config: RidgeForecastConfig | None = None) -> None:
        self.config = config or RidgeForecastConfig()
        self.config.validate()
        self.regressor = RidgeRegressor(self.config.alpha)
        self.training_samples = 0

    @staticmethod
    def _calendar(timestamp: np.datetime64) -> np.ndarray:
        hour = float(timestamp.astype("datetime64[h]").astype(np.int64) % 24)
        year_start = timestamp.astype("datetime64[Y]")
        day = float((timestamp.astype("datetime64[D]") - year_start).astype(int))
        return np.array(
            [
                np.sin(2.0 * np.pi * hour / 24.0),
                np.cos(2.0 * np.pi * hour / 24.0),
                np.sin(2.0 * np.pi * day / 365.2425),
                np.cos(2.0 * np.pi * day / 365.2425),
            ],
            dtype=np.float64,
        )

    def design_row(
        self,
        history: np.ndarray,
        origin_time: np.datetime64,
    ) -> np.ndarray:
        """Create the exact feature vector used in fitting and evaluation."""
        values = np.asarray(history, dtype=np.float64)
        if values.shape != (self.config.history_length,):
            raise ValueError(
                f"history must contain exactly {self.config.history_length} values"
            )
        if not np.isfinite(values).all():
            raise ValueError("forecast history must be finite")
        if self.config.include_calendar:
            return np.concatenate((values, self._calendar(origin_time)))
        return values.copy()

    def fit_series(
        self,
        values: np.ndarray,
        times: np.ndarray,
        train_end: int,
    ) -> AutoregressiveRidgeForecaster:
        """Fit only on origins and targets strictly before ``train_end``."""
        series = np.asarray(values, dtype=np.float64)
        timestamps = np.asarray(times).astype("datetime64[ns]")
        if series.ndim != 1 or timestamps.ndim != 1 or len(series) != len(timestamps):
            raise ValueError("values and times must be aligned one-dimensional arrays")
        if not self.config.history_length < train_end <= len(series):
            raise ValueError("train_end leaves no valid autoregressive training region")
        if len(timestamps) > 1 and np.any(np.diff(timestamps.astype(np.int64)) <= 0):
            raise ValueError("forecast timestamps must be strictly increasing")

        rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for origin in complete_origins(
            series,
            history_length=self.config.history_length,
            horizons=self.config.horizons,
            start=0,
            end=train_end,
        ):
            history = series[origin - self.config.history_length + 1 : origin + 1]
            target = series[origin + np.asarray(self.config.horizons)]
            rows.append(self.design_row(history, timestamps[int(origin)]))
            targets.append(target)
        if len(rows) < 2:
            raise ValueError(
                "clean training split contains fewer than two valid windows"
            )
        self.regressor.fit(np.stack(rows), np.stack(targets))
        self.training_samples = len(rows)
        return self

    def predict_history(
        self,
        history: np.ndarray,
        origin_time: np.datetime64,
    ) -> np.ndarray:
        """Forecast every configured horizon from one repaired history."""
        return self.regressor.predict(self.design_row(history, origin_time))[0]

    def get_config(self) -> dict[str, int | float | bool | tuple[int, ...]]:
        """Return the complete reproducible forecaster configuration."""
        return asdict(self.config)
