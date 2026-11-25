"""Decoding, units and calibrated donor correction."""

import numpy as np
import torch

PHYSICAL_BOUNDS_BY_VARIABLE: dict[str, tuple[float, float]] = {
    "temperature": (-80.0, 60.0),
    "dewpoint": (-90.0, 50.0),
    "precipitation": (0.0, 300.0),
    "humidity": (0.0, 100.0),
    "pressure": (550.0, 1100.0),
    "wind_speed": (0.0, 75.0),
    "wind_direction": (0.0, 360.0),
}


UNITS_BY_VARIABLE = {
    "temperature": "degC",
    "dewpoint": "degC",
    "precipitation": "mm",
    "humidity": "%",
    "pressure": "hPa",
    "wind_speed": "m s-1",
    "wind_direction": "degree",
}


def physical_prediction(
    prediction: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    *,
    variable: str,
    constrain: bool = True,
) -> torch.Tensor:
    """Decode raw network outputs."""
    bounds = PHYSICAL_BOUNDS_BY_VARIABLE[variable]
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError("Prediction scales must be finite and positive")
    physical = center[:, None, :] + prediction * scale[:, None, :]
    if not bool(torch.isfinite(physical).all()):
        raise ValueError("Non-finite model predictions cannot be postprocessed")
    if variable == "wind_direction":
        return torch.remainder(physical, 360.0)
    return physical.clamp(*bounds) if constrain else physical


def donor_correction(
    sample: dict[str, torch.Tensor], p: np.ndarray, offsets: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    n = len(p)
    vis = (
        sample["input_mask"][:n, 1:, 3]
        & sample["availability"][:n, 1:, 3]
        & sample["node_mask"][None, 1:]
    ).numpy()
    amount = (
        (
            sample["values"][:n, 1:, 3] * sample["scale"][None, 1:, 3]
            + sample["center"][None, 1:, 3]
        )
        .clamp(0, 300)
        .double()
        .numpy()
    )
    count = vis.sum(1)
    wet = (vis & (amount >= 0.1)).sum(1)
    mean = np.where(vis, amount, 0).sum(1) / np.maximum(count, 1)
    bins = np.digitize(p, [0.1, 1, 5])
    fraction = (3 * wet >= count).astype(int) + (3 * wet >= 2 * count).astype(int)
    detail = 4 + 12 * bins + 4 * fraction + np.digitize(mean, [0.1, 1, 5])
    shift = np.array(
        [
            offsets.get(str(j), offsets.get(str(i), 0.0))
            for i, j in zip(bins, detail, strict=True)
        ]
    )
    use = (count >= 3) & np.isfinite(p)
    return np.where(use, np.clip(p + shift, 0, 300), p), use


def humidity_proxy(times, fields, coef):
    axis = times.astype("datetime64[h]").astype(np.int64)
    season = 2 * np.pi * times.astype("datetime64[D]").astype(np.int64) / 365.25
    phase = 2 * np.pi * (axis % 24) / 24
    x = np.column_stack(
        (
            np.ones(len(axis)),
            np.sin(phase),
            np.cos(phase),
            np.sin(season),
            np.cos(season),
            fields["t2m"] - 273.15,
            np.log1p(np.maximum(0, fields["tp"] * 1000)),
            fields["sp"] / 100,
            fields["u10"],
            fields["v10"],
            np.hypot(fields["u10"], fields["v10"]),
        )
    )
    return np.clip(x @ coef, 0, 100)
