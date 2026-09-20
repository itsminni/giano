"""Simple downstream models used to measure imputation utility."""

from giano.downstream.ridge_forecaster import (
    AutoregressiveRidgeForecaster,
    RidgeForecastConfig,
)

__all__ = [
    "AutoregressiveRidgeForecaster",
    "RidgeForecastConfig",
]
