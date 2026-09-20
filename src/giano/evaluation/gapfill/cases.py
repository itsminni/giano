"""Canonical benchmark cases shared by every gap-filling model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

from giano.evaluation.gapfill.masks import SyntheticMaskType, empirical_block_lengths
from giano.experiments import experiment_seeds
from giano.spatiotemporal_dataset import (
    SpatiotemporalPanel,
    SpatiotemporalWindowDataset,
    TemporalSplit,
)

if TYPE_CHECKING:
    from giano.baselines.train_bilstm import BiLSTMTrainingConfig
    from giano.model.train_imputeformer import TrainingConfig


@dataclass(frozen=True)
class BenchmarkCase:
    """One frozen missingness pattern in the common benchmark suite."""

    mask_type: SyntheticMaskType
    parameter: int | float

    def __post_init__(self) -> None:
        if self.mask_type == "point":
            if not 0 < float(self.parameter) < 1:
                raise ValueError("point case parameter must be inside (0, 1)")
        elif self.mask_type == "empirical":
            if not 0 < float(self.parameter) <= 1:
                raise ValueError("empirical parameter must be a quantile in (0, 1]")
        elif int(self.parameter) <= 0 or float(self.parameter) != int(self.parameter):
            raise ValueError("block case parameter must be a positive hour count")

    @property
    def parameter_unit(self) -> str:
        """Return the machine-readable unit of ``parameter``."""
        if self.mask_type == "point":
            return "fraction"
        return "quantile" if self.mask_type == "empirical" else "hours"

    @property
    def label(self) -> str:
        """Return a compact label for tables and logs."""
        if self.mask_type == "point":
            return f"point-{float(self.parameter):.0%}"
        if self.mask_type == "empirical":
            return f"empirical-train-q{float(self.parameter):.0%}"
        return f"{self.mask_type}-{int(self.parameter)}h"


STANDARD_GAPFILL_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase("point", 0.1),
    BenchmarkCase("point", 0.3),
    BenchmarkCase("point", 0.5),
    BenchmarkCase("block", 3),
    BenchmarkCase("block", 6),
    BenchmarkCase("block", 12),
    BenchmarkCase("block", 24),
    BenchmarkCase("spatial_block", 6),
    BenchmarkCase("spatial_block", 12),
    BenchmarkCase("terminal", 6),
    BenchmarkCase("terminal", 12),
    BenchmarkCase("terminal", 24),
    BenchmarkCase("empirical", 0.95),
)


def dataset_for_case(
    panel: SpatiotemporalPanel,
    split: str,
    config: TrainingConfig | BiLSTMTrainingConfig,
    case: BenchmarkCase,
    seed: int,
) -> SpatiotemporalWindowDataset:
    """Build identical windows and masks for every model and reconstruction export."""
    if case.mask_type == "empirical":
        block_lengths = empirical_block_lengths(
            panel.values,
            panel.coverage_bounds,
            int(len(panel.times) * 0.7),
            quantile_cap=float(case.parameter),
            max_length=config.seq_len,
        )
    else:
        block_lengths = (int(case.parameter) if case.mask_type != "point" else 3,)
    point_rate = float(case.parameter) if case.mask_type == "point" else 0.3
    return SpatiotemporalWindowDataset(
        panel,
        cast(TemporalSplit, split),
        seq_len=config.seq_len,
        stride=config.seq_len,
        max_nodes=config.max_nodes,
        min_context_points=config.min_context_points,
        seed=seed,
        mask_mode=case.mask_type,
        point_rate=(point_rate, point_rate),
        block_lengths=block_lengths,
    )


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Configuration that makes an evaluation run reproducible."""

    split: str
    seeds: tuple[int, ...]
    max_batches: int
    cases: tuple[BenchmarkCase, ...]

    def validate(self) -> None:
        """Reject incomplete or scientifically ambiguous protocols."""
        if self.split not in {"val", "test"}:
            raise ValueError("benchmark split must be val or test")
        if not self.seeds:
            raise ValueError("benchmark must contain at least one seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("benchmark seeds must be unique")
        if self.max_batches <= 0:
            raise ValueError("benchmark max_batches must be positive")
        if not self.cases:
            raise ValueError("benchmark must contain at least one case")
        if len(set(self.cases)) != len(self.cases):
            raise ValueError("benchmark cases must be unique")


def protocol_with_overrides(
    base: BenchmarkProtocol,
    args: argparse.Namespace,
) -> BenchmarkProtocol:
    """Apply the same CLI overrides to every model's benchmark protocol."""
    if args.seeds is not None and args.training_seeds is not None:
        raise ValueError(
            "--seeds varies masks for one checkpoint and cannot be combined "
            "with --training-seeds"
        )
    requested_seeds = (
        args.training_seeds
        if args.training_seeds is not None
        else args.seeds
        if args.seeds is not None
        else base.seeds
    )
    protocol = replace(
        base,
        split=args.split or base.split,
        seeds=experiment_seeds(base.seeds[0], requested_seeds),
        max_batches=(
            args.max_batches if args.max_batches is not None else base.max_batches
        ),
    )
    protocol.validate()
    return protocol


def _case_from_mapping(raw: Any) -> BenchmarkCase:
    if not isinstance(raw, dict):
        raise ValueError("each benchmark case must be a mapping")
    mask_type = raw.get("mask_type")
    parameter = raw.get("parameter")
    if mask_type not in {
        "point",
        "block",
        "spatial_block",
        "terminal",
        "empirical",
    }:
        raise ValueError(f"unsupported benchmark mask_type: {mask_type}")
    if not isinstance(parameter, (int, float)) or isinstance(parameter, bool):
        raise ValueError("benchmark case parameter must be numeric")
    return BenchmarkCase(mask_type, parameter)


def load_benchmark_protocol(path: Path) -> BenchmarkProtocol:
    """Load the frozen benchmark protocol from the sole project YAML file."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration top-level must be a mapping")
    section = raw.get("benchmark", {})
    if not isinstance(section, dict):
        raise ValueError("benchmark configuration must be a mapping")
    raw_cases = section.get("cases")
    cases = (
        tuple(_case_from_mapping(item) for item in raw_cases)
        if isinstance(raw_cases, list)
        else STANDARD_GAPFILL_CASES
    )
    raw_seeds = section.get("seeds", [42])
    if not isinstance(raw_seeds, list) or not all(
        isinstance(seed, int) and not isinstance(seed, bool) for seed in raw_seeds
    ):
        raise ValueError("benchmark seeds must be a list of integers")
    protocol = BenchmarkProtocol(
        split=str(section.get("split", "test")),
        seeds=tuple(raw_seeds),
        max_batches=int(section.get("max_batches", 200)),
        cases=cases,
    )
    protocol.validate()
    return protocol
