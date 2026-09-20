"""Shared physical-unit decoding for evaluation, inference and ONNX clients."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from giano.meteorology import PHYSICAL_BOUNDS_BY_VARIABLE

PREDICTION_POLICY_VERSION = "physical-bounds-v1"


def prediction_policy() -> dict[str, Any]:
    """Describe postprocessing explicitly in artifacts, not in network weights."""
    return {
        "version": PREDICTION_POLICY_VERSION,
        "scalar_bounds": {
            k: list(v)
            for k, v in PHYSICAL_BOUNDS_BY_VARIABLE.items()
            if k != "wind_direction"
        },
        "wind_direction": "modulo_360",
        "nonfinite_predictions": "error",
        "source_observations": "preserved",
    }


def physical_prediction(
    prediction: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    *,
    variable: str,
    constrain: bool = True,
) -> torch.Tensor:
    """Decode raw network outputs; constrain predictions, never observations."""
    bounds = PHYSICAL_BOUNDS_BY_VARIABLE[variable]
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError("Prediction scales must be finite and positive")
    physical = center[:, None, :] + prediction * scale[:, None, :]
    if not bool(torch.isfinite(physical).all()):
        raise ValueError("Non-finite model predictions cannot be postprocessed")
    if variable == "wind_direction":
        return torch.remainder(physical, 360.0)
    return physical.clamp(*bounds) if constrain else physical


def physical_prediction_numpy(
    prediction: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    variable: str,
    constrain: bool = True,
) -> np.ndarray:
    """Apply the same contract to normalized ONNX output outside the core."""
    bounds = PHYSICAL_BOUNDS_BY_VARIABLE[variable]
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Prediction scales must be finite and positive")
    physical = center[:, None, :] + prediction * scale[:, None, :]
    if not np.isfinite(physical).all():
        raise ValueError("Non-finite model predictions cannot be postprocessed")
    if variable == "wind_direction":
        return np.remainder(physical, 360.0)
    return np.clip(physical, *bounds) if constrain else physical
