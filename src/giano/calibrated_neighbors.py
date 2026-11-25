"""Estimate a missing station series from simultaneous, historically calibrated donors."""

import numpy as np

from .model import VARIABLES
from .physical import PHYSICAL_BOUNDS_BY_VARIABLE
from .windows import LongPanel, Outage, nearest_nodes

HOUR = np.timedelta64(1, "h")
FIELDS = ("temperature", "wind_speed", "wind_direction")


def _channels(values, circular):
    if not circular:
        return values
    radians = np.deg2rad(values)
    return np.column_stack((np.sin(radians), np.cos(radians)))


def _design(values, times, circular):
    phase = (times.astype("datetime64[h]").astype(np.int64) % 24) * (2 * np.pi / 24)
    return np.column_stack((_channels(values, circular), np.sin(phase), np.cos(phase)))


def calibrated_neighbors(panel: LongPanel, gap: Outage, variable: str) -> np.ndarray:
    """Fit on at most one year before the full outage; NaN marks lack of support."""
    result = np.full(gap.end - gap.start, np.nan)
    if variable not in FIELDS or len(result) < 168 or gap.regime != "other-stations":
        return result
    col = VARIABLES.index(variable)
    left = int(panel.times.searchsorted(panel.times[gap.start] - 365 * 24 * HOUR))
    if gap.start - left < 720:
        return result
    history = np.where(
        panel.availability[left : gap.start, :, col],
        panel.values[left : gap.start, :, col],
        np.nan,
    ).astype(float)
    target = history[:, gap.node]
    order = nearest_nodes(panel, gap.node, len(panel.station_ids))[1:]
    pairs = np.isfinite(target[:, None]) & np.isfinite(history[:, order])
    donors = order[pairs.sum(axis=0) >= 720][:8]
    if not len(donors):
        return result
    past = history[:, donors]
    use = np.isfinite(target) & np.isfinite(past).any(axis=1)
    circular = variable == "wind_direction"
    x = _design(past[use], panel.times[left : gap.start][use], circular)
    center = np.nanmean(x, axis=0)
    x = np.nan_to_num(x - center)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1
    x /= scale
    y = _channels(target[use, None], circular)
    mean = y.mean(axis=0)
    coef = np.linalg.solve(x.T @ x + 24 * np.eye(x.shape[1]), x.T @ (y - mean))
    sl = slice(gap.start, gap.end)
    current = np.where(
        panel.availability[sl, :, col][:, donors],
        panel.values[sl, :, col][:, donors],
        np.nan,
    )
    active = np.isfinite(current).any(axis=1)
    design = _design(current[active], panel.times[sl][active], circular)
    estimate = np.nan_to_num((design - center) / scale) @ coef + mean
    if circular:
        sine, cosine = estimate.T
        result[active] = np.where(
            np.hypot(sine, cosine) > 1e-6,
            np.rad2deg(np.arctan2(sine, cosine)) % 360,
            np.nan,
        )
    else:
        result[active] = estimate[:, 0].clip(*PHYSICAL_BOUNDS_BY_VARIABLE[variable])
    return result
