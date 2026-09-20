"""Single implementation of synthetic masks used across Giano models."""

from __future__ import annotations

from typing import Literal

import numpy as np

SyntheticMaskType = Literal[
    "point",
    "block",
    "spatial_block",
    "terminal",
    "empirical",
]


def natural_gap_lengths(
    values: np.ndarray,
    coverage_bounds: np.ndarray,
    end: int,
) -> np.ndarray:
    """Return natural missing-run lengths inside coverage before ``end``."""
    observations = np.asarray(values)
    bounds = np.asarray(coverage_bounds, dtype=np.int64)
    if observations.ndim != 2 or bounds.shape != (observations.shape[1], 2):
        raise ValueError("values and coverage_bounds are not aligned")
    if not 0 < end <= observations.shape[0]:
        raise ValueError("natural-gap end must lie inside the time axis")
    lengths: list[int] = []
    for node in range(observations.shape[1]):
        lower = max(0, int(bounds[node, 0]))
        upper = min(end, int(bounds[node, 1]) + 1)
        if upper <= lower:
            continue
        missing = ~np.isfinite(observations[lower:upper, node])
        padded = np.pad(missing.astype(np.int8), (1, 1))
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        lengths.extend(
            int(stop - start) for start, stop in zip(starts, stops, strict=True)
        )
    return np.asarray(lengths, dtype=np.int64)


def empirical_block_lengths(
    values: np.ndarray,
    coverage_bounds: np.ndarray,
    train_end: int,
    *,
    quantile_cap: float,
    max_length: int,
    samples: int = 256,
) -> tuple[int, ...]:
    """Approximate the train-only gap distribution with deterministic quantiles."""
    if not 0 < quantile_cap <= 1:
        raise ValueError("empirical quantile cap must be inside (0, 1]")
    if max_length <= 1 or samples <= 0:
        raise ValueError("empirical mask limits must be positive")
    lengths = natural_gap_lengths(values, coverage_bounds, train_end)
    lengths = lengths[lengths > 0]
    if not lengths.size:
        raise ValueError("training split contains no usable natural gaps")
    upper = max(
        1, min(int(np.ceil(np.quantile(lengths, quantile_cap))), max_length - 1)
    )
    capped = np.clip(lengths, 1, upper)
    probabilities = np.linspace(0.0, 1.0, min(samples, len(capped)))
    representatives = np.rint(np.quantile(capped, probabilities)).astype(np.int64)
    representatives = np.clip(representatives, 1, max_length - 1)
    return tuple(int(item) for item in representatives)


def _validate_observed(observed: np.ndarray) -> np.ndarray:
    values = np.asarray(observed, dtype=bool)
    if values.ndim != 2:
        raise ValueError("observed must have shape [time, nodes]")
    if values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("observed must contain at least two times and one node")
    return values


def point_mask(
    observed: np.ndarray,
    rng: np.random.Generator,
    rate: float,
    *,
    min_visible: int = 2,
) -> np.ndarray:
    """Hide independent observed points while preserving visible context."""
    values = _validate_observed(observed)
    if not 0 < rate < 1:
        raise ValueError("point mask rate must be inside (0, 1)")
    if min_visible < 1:
        raise ValueError("min_visible must be positive")
    mask = values & (rng.random(values.shape) < rate)
    for node in range(values.shape[1]):
        visible = np.flatnonzero(values[:, node] & ~mask[:, node])
        if visible.size >= min_visible:
            continue
        candidates = np.flatnonzero(mask[:, node])
        restore = candidates[: min(min_visible - visible.size, candidates.size)]
        mask[restore, node] = False
    return mask


def block_mask(
    observed: np.ndarray,
    rng: np.random.Generator,
    length: int,
    *,
    shared_time: bool,
    terminal: bool = False,
    min_visible: int = 2,
) -> np.ndarray:
    """Hide temporal blocks, optionally shared across a subset of stations."""
    values = _validate_observed(observed)
    if not 0 < length < values.shape[0]:
        raise ValueError("block length must be smaller than the time dimension")
    if min_visible < 1:
        raise ValueError("min_visible must be positive")

    mask = np.zeros_like(values)
    shared_start = (
        values.shape[0] - length
        if terminal
        else int(rng.integers(0, values.shape[0] - length + 1))
    )
    nodes = np.arange(values.shape[1])
    if shared_time and nodes.size > 1:
        keep = max(1, int(np.ceil(nodes.size * float(rng.uniform(0.25, 0.75)))))
        nodes = rng.choice(nodes, keep, replace=False)
    for node in nodes:
        start = (
            shared_start
            if shared_time or terminal
            else int(rng.integers(0, values.shape[0] - length + 1))
        )
        candidate = values[start : start + length, node]
        if values[:, node].sum() - candidate.sum() >= min_visible:
            mask[start : start + length, node] = candidate
    return mask


def synthetic_mask(
    observed: np.ndarray,
    rng: np.random.Generator,
    mask_type: SyntheticMaskType,
    parameter: int | float,
    *,
    empirical_lengths: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Materialize one explicit mask specification on an observed panel."""
    if mask_type == "point":
        return point_mask(observed, rng, float(parameter))
    if mask_type == "empirical":
        if not empirical_lengths:
            raise ValueError("empirical mask requires train-only gap lengths")
        length = int(rng.choice(empirical_lengths))
    else:
        length = int(parameter)
    if mask_type != "empirical" and float(parameter) != length:
        raise ValueError("block mask parameters must be integer hour counts")
    return block_mask(
        observed,
        rng,
        length,
        shared_time=mask_type in {"spatial_block", "terminal"},
        terminal=mask_type == "terminal",
    )
