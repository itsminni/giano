"""Calendar features and pooled standardised Ridge for the paired studies."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

HORIZONS = (1, 6, 12, 24)


def calendar(times):
    hour = times.astype("datetime64[h]").astype(int) % 24
    day = (times.astype("datetime64[D]") - times.astype("datetime64[Y]")).astype(int)
    return np.column_stack(
        (
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day / 365.2425),
            np.cos(2 * np.pi * day / 365.2425),
        )
    )


def pooled_fit(parts: Sequence[tuple[int, NDArray[np.float64], NDArray[np.float64]]]):
    """Fit the same standardised Ridge to pooled rows using their centred cross-products."""
    n = sum(size for size, _, _ in parts)
    if n == 0:
        raise ValueError("Pooled Ridge requires at least one training window")
    mean = np.zeros_like(parts[0][1])
    for size, center, _ in parts:
        mean += size * center
    mean /= n

    scatter = np.zeros_like(parts[0][2])
    for size, center, within in parts:
        offset = center - mean
        scatter += within + size * np.outer(offset, offset)
    nx = len(mean) - len(HORIZONS)
    gram = scatter[:nx, :nx]
    scale = np.sqrt(np.maximum(0, gram.diagonal() / n))
    # Convert the standardised alpha=1 penalty into raw predictor units.
    penalty = np.where(scale > 1e-12, scale, 1.0) ** 2
    beta = np.linalg.solve(gram + np.diag(penalty), scatter[:nx, nx:])
    return beta, mean[nx:] - mean[:nx] @ beta, n
