"""Shared experiment-seed and checkpoint-path conventions."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

MAX_EXPERIMENT_SEED = 2**32 - 1


def experiment_seeds(
    default_seed: int,
    requested_seeds: Iterable[int] | None,
) -> tuple[int, ...]:
    """Return validated, unique seeds while preserving the requested order."""
    seeds = tuple(requested_seeds) if requested_seeds is not None else (default_seed,)
    if not seeds:
        raise ValueError("at least one experiment seed is required")
    if any(
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= MAX_EXPERIMENT_SEED
        for seed in seeds
    ):
        raise ValueError(
            f"experiment seeds must be integers between 0 and {MAX_EXPERIMENT_SEED}"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError("experiment seeds must be unique")
    return seeds


def imputeformer_checkpoint_path(
    checkpoint_dir: Path,
    variable: str,
    *,
    seed: int | None,
) -> Path:
    """Return a canonical or seed-specific ImputeFormer checkpoint path."""
    if seed is None:
        return checkpoint_dir / f"{variable}.pt"
    return checkpoint_dir / f"seed-{seed}" / f"{variable}.pt"


def bilstm_checkpoint_path(
    checkpoint_dir: Path,
    variant: str,
    variable: str,
    *,
    seed: int | None,
) -> Path:
    """Return a canonical or seed-specific BiLSTM checkpoint path."""
    if seed is None:
        return checkpoint_dir / variant / f"{variable}.pt"
    return checkpoint_dir / variant / f"seed-{seed}" / f"{variable}.pt"
