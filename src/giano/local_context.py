"""Local observations available to a retrospective or origin-bounded repair."""

import numpy as np

from .model import VARIABLES
from .physical import PHYSICAL_BOUNDS_BY_VARIABLE
from .windows import LongPanel, Outage, nearest_nodes

HOUR = np.timedelta64(1, "h")
FIELDS = ("temperature", "humidity", "pressure")


def local_estimates(
    panel: LongPanel,
    gap: Outage,
    variable: str,
    *,
    hidden: np.ndarray | None = None,
    donor_history_end: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Use visible station boundaries and donor offsets from the preceding week.

    The optional mask describes holes inside a rolling forecast history. Removing
    them all up front keeps earlier hidden readings out of later donor fits.
    """
    col = VARIABLES.index(variable)
    length = gap.end - gap.start
    removed = np.ones(length, dtype=bool) if hidden is None else hidden
    if donor_history_end is not None and (
        hidden is not None or not 0 <= donor_history_end <= gap.start
    ):
        raise ValueError(
            "A natural gap needs a reference ending at or before its start"
        )
    if removed.shape != (length,) or removed.dtype != bool:
        raise ValueError("Expected one boolean per candidate hour")
    bridge = np.full(length, np.nan)
    spatial = np.full(length, np.nan)
    if variable not in FIELDS:
        return bridge, spatial
    clock = panel.times
    first = clock.searchsorted(clock[gap.start] - 168 * HOUR)
    stop = clock.searchsorted(clock[gap.end - 1] + 7 * HOUR)
    times = clock[first:stop].astype("datetime64[h]").astype(np.int64)

    def visible_values(lo, hi):
        return np.where(
            panel.availability[lo:hi, :, col], panel.values[lo:hi, :, col], np.nan
        )

    values = visible_values(first, stop)
    node = gap.node
    start = gap.start - first
    edges = np.flatnonzero(np.diff(removed, prepend=False, append=False)).reshape(-1, 2)
    values[start + np.flatnonzero(removed), node] = np.nan
    target = values[:, node]
    visible = np.flatnonzero(np.isfinite(target))
    order = nearest_nodes(panel, node, len(panel.station_ids))[1:]
    for left, right in edges:
        begin, end = start + left, start + right
        pos = visible.searchsorted(begin)
        if 0 < pos < len(visible):
            before, after = visible[pos - 1 : pos + 1]
            elapsed = times[after] - times[before]
            # Six missing hours span seven hours between the bounding readings.
            if elapsed <= 7:
                bridge[left:right] = np.interp(
                    times[begin:end], times[[before, after]], target[[before, after]]
                )
        if gap.regime != "other-stations" or not len(order):
            continue
        if donor_history_end is None:
            history = values[np.searchsorted(times, times[begin] - 168) : begin]
        else:
            # A file request can start partway through a natural gap.
            lo = clock.searchsorted(clock[donor_history_end] - 168 * HOUR)
            history = visible_values(lo, donor_history_end)
        diff = history[:, node, None] - history[:, order]
        eligible = np.flatnonzero(np.isfinite(diff).sum(axis=0) >= 24)[:8]
        if not len(eligible):
            continue
        offsets = np.nanmedian(diff[:, eligible], axis=0)
        adjusted = values[begin:end, order[eligible]] + offsets
        active = np.isfinite(adjusted).any(axis=1)
        spatial[left + np.flatnonzero(active)] = np.nanmedian(adjusted[active], axis=1)
    return bridge, spatial.clip(*PHYSICAL_BOUNDS_BY_VARIABLE[variable])


def apply_local_context(
    panel: LongPanel,
    gap: Outage,
    variable: str,
    prediction: np.ndarray,
    *,
    hidden: np.ndarray | None = None,
    allow_bridge: bool = True,
    donor_history_end: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep ensemble coverage and report the source of each local replacement."""
    bridge, spatial = local_estimates(
        panel, gap, variable, hidden=hidden, donor_history_end=donor_history_end
    )
    supported = np.isfinite(prediction)
    use_spatial = supported & np.isfinite(spatial)
    use_bridge = supported & np.isfinite(bridge) & allow_bridge
    updated = np.where(use_bridge, bridge, np.where(use_spatial, spatial, prediction))
    source = np.where(use_bridge, 5, np.where(use_spatial, 1, 0)).astype(np.uint8)
    return updated, source
