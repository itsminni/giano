"""Optional station-level checks for pressure and short wind-direction gaps."""

import numpy as np

from .model import VARIABLES
from .physical import PHYSICAL_BOUNDS_BY_VARIABLE
from .windows import LongPanel, Outage


def station_reference(
    variable: str,
    before: np.ndarray,
    after: float,
    estimate: np.ndarray,
    duration: int,
    offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the week before the true gap and the first reading after it, if visible.

    The hourly input may contain NaNs. Only already supported hours can change.
    Offset and duration refer to the entire gap when filling a requested slice.
    """
    updated = estimate.copy()
    source = np.zeros(len(estimate), np.uint8)
    supported = np.isfinite(estimate)
    if variable == "pressure":
        bottom, top = PHYSICAL_BOUNDS_BY_VARIABLE["pressure"]
        if (
            7 <= duration <= 12
            and len(before)
            and bottom <= before[-1] <= top
            and bottom <= after <= top
        ):
            fraction = (offset + np.arange(1, len(estimate) + 1)) / (duration + 1)
            updated[supported] = before[-1] + (after - before[-1]) * fraction[supported]
            source[supported] = 5
            return updated, source
        history = before[-168:]
        history = history[(history >= bottom) & (history <= top)]
        if len(history) >= 24:
            low, level, high = np.percentile(history, [25, 50, 75])
            band = max(20.0, 4 * (high - low) / 1.349)
            replace = supported & (np.abs(estimate - level) > band)
            updated[replace], source[replace] = level, 3
    elif variable == "wind_direction" and duration <= 12:
        left = before[-1] if len(before) else np.nan
        has_left, has_right = 0 <= left <= 360, 0 <= after <= 360
        if has_left and has_right:
            arc = (after - left + 180) % 360 - 180
            if not np.isclose(abs(arc), 180):
                fraction = (offset + np.arange(1, len(estimate) + 1)) / (duration + 1)
                updated[supported] = (left + arc * fraction[supported]) % 360
                source[supported] = 5
        elif has_left or has_right:
            updated[supported] = (left if has_left else after) % 360
            source[supported] = 6
    return updated, source


def apply_station_reference(
    panel: LongPanel,
    gap: Outage,
    variable: str,
    prediction: np.ndarray,
    *,
    hidden: np.ndarray | None = None,
    offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Read only visible station hours, including for multiple holes in one history."""
    updated = prediction.copy()
    source = np.zeros(len(prediction), dtype=np.uint8)
    if variable not in ("pressure", "wind_direction"):
        return updated, source
    hour = np.timedelta64(1, "h")
    col = VARIABLES.index(variable)
    clock = panel.times
    if hidden is None:
        runs = np.array([[gap.start, gap.end]])
    else:
        if (
            offset
            or hidden.dtype != bool
            or hidden.shape != prediction.shape
            or len(hidden) != gap.end - gap.start
        ):
            raise ValueError("A rolling station reference requires the complete mask")
        runs = gap.start + np.flatnonzero(
            np.diff(hidden, prepend=False, append=False)
        ).reshape(-1, 2)
    for start, end in runs:
        lo = int(clock.searchsorted(clock[start] - 168 * hour))
        if hidden is not None:
            lo = max(lo, gap.start)
        before = np.where(
            panel.availability[lo:start, gap.node, col],
            panel.values[lo:start, gap.node, col],
            np.nan,
        )
        if hidden is not None:
            before[hidden[lo - gap.start : start - gap.start]] = np.nan
        if len(before) and clock[start] - clock[start - 1] != hour:
            before = np.array([], dtype=float)
        after = np.nan
        if (
            end < (len(clock) if hidden is None else gap.end)
            and clock[end] - clock[end - 1] == hour
            and panel.availability[end, gap.node, col]
        ):
            after = float(panel.values[end, gap.node, col])
        i = start - gap.start
        rows = slice(i, i + end - start)
        updated[rows], source[rows] = station_reference(
            variable, before, after, prediction[rows], end - start, offset
        )
    return updated, source
