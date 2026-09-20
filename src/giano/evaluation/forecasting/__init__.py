"""Leakage-safe downstream forecasting evaluation protocols."""

from giano.evaluation.forecasting.protocol import (
    RollingOriginConfig,
    evaluate_rolling_origin,
    load_forecasting_config,
    paired_method_comparisons,
    summarize_forecasts,
)

__all__ = [
    "RollingOriginConfig",
    "evaluate_rolling_origin",
    "load_forecasting_config",
    "paired_method_comparisons",
    "summarize_forecasts",
]
