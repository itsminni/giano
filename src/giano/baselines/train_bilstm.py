"""Train historical-style and fair BiLSTM baselines in pure PyTorch."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from giano.baselines.bilstm import (
    BiLSTMConfig,
    BiLSTMVariant,
    MeteorologicalBiLSTM,
)
from giano.evaluation.gapfill.results import (
    EvaluationResult,
    evaluate_model,
    normalized_error,
    to_device,
)
from giano.experiments import bilstm_checkpoint_path, experiment_seeds
from giano.prediction import prediction_policy
from giano.provenance import training_dataset_identity, validate_training_dataset
from giano.runtime import default_device
from giano.spatiotemporal_dataset import (
    MaskMode,
    SpatiotemporalPanel,
    SpatiotemporalWindowDataset,
    build_spatiotemporal_panel,
)
from giano.training_progress import (
    atomic_torch_save,
    capture_rng_state,
    load_training_progress,
    progress_checkpoint_path,
    restore_rng_state,
)
from giano.variables import VARIABLE_TYPE_NAMES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BiLSTMTrainingConfig:
    """Shared split, window, optimization, and masking settings."""

    seq_len: int = 72
    stride: int = 12
    max_nodes: int = 24
    min_context_points: int = 4
    batch_size: int = 8
    num_workers: int = 0
    max_epochs: int = 12
    max_batches_per_epoch: int | None = None
    max_validation_batches: int = 100
    learning_rate: float = 1e-3
    legacy_learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 4
    gradient_clip: float = 1.0
    seed: int = 42
    point_rate_min: float = 0.15
    point_rate_max: float = 0.45
    legacy_point_rate: float = 0.5
    block_lengths: tuple[int, ...] = (3, 6, 12, 24)

    def validate(self) -> None:
        """Reject invalid experiments before reading the dataset."""
        positive = (
            self.seq_len,
            self.stride,
            self.max_nodes,
            self.min_context_points,
            self.batch_size,
            self.max_epochs,
            self.max_validation_batches,
            self.patience,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("BiLSTM training dimensions must be positive")
        if self.max_batches_per_epoch is not None and self.max_batches_per_epoch <= 0:
            raise ValueError("max_batches_per_epoch must be positive or null")
        if not 0 < self.point_rate_min <= self.point_rate_max < 1:
            raise ValueError("Invalid fair point-mask range")
        if not 0 < self.legacy_point_rate < 1:
            raise ValueError("Invalid legacy point-mask rate")
        if min(self.learning_rate, self.legacy_learning_rate) <= 0:
            raise ValueError("BiLSTM learning rates must be positive")


def read_bilstm_config(
    path: Path,
    variant: BiLSTMVariant,
) -> tuple[BiLSTMConfig, BiLSTMTrainingConfig]:
    """Read a variant from the sole application YAML configuration."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration top-level must be a mapping")
    section = raw.get("bilstm", {})
    if not isinstance(section, dict):
        raise ValueError("bilstm configuration must be a mapping")
    model_raw = section.get("model", {})
    training_raw = section.get("training", {})
    if not isinstance(model_raw, dict) or not isinstance(training_raw, dict):
        raise ValueError("bilstm model/training sections must be mappings")
    model_values = {
        key: value
        for key, value in model_raw.items()
        if key in BiLSTMConfig.__dataclass_fields__ and key != "variant"
    }
    training_values = {
        key: value
        for key, value in training_raw.items()
        if key in BiLSTMTrainingConfig.__dataclass_fields__
    }
    if "block_lengths" in training_values:
        training_values["block_lengths"] = tuple(training_values["block_lengths"])
    model_config = BiLSTMConfig(variant=variant, **model_values)
    training_config = BiLSTMTrainingConfig(**training_values)
    model_config.validate()
    training_config.validate()
    return model_config, training_config


def _recipe(variant: BiLSTMVariant) -> dict[str, Any]:
    if variant == "legacy_retrained":
        return {
            "historical_source": "challenge1_gapfill",
            "historical_source_access": "read_only",
            "input_channels": ["station_value", "auxiliary_value"],
            "missing_value": -100.0,
            "training_mask": "independent_points_50pct",
            "training_loss": "mse_all_originally_observed_points",
            "intentional_differences": [
                "current hourly NetCDF contract",
                "current time-disjoint splits",
                "72-hour common context instead of historical 24-hour context",
                "current per-window normalization before inserting the sentinel",
                "same-mask hidden-only benchmark metrics",
                "station-only variables use a neutral unavailable auxiliary channel",
            ],
        }
    return {
        "input_channels": "all_common_features_plus_coordinates",
        "missing_value": 0.0,
        "explicit_observation_mask": True,
        "training_mask": "same_mixture_as_giano",
        "training_loss": "mae_synthetically_hidden_points_only",
    }


def _make_dataset(
    panel: SpatiotemporalPanel,
    split: str,
    config: BiLSTMTrainingConfig,
    variant: BiLSTMVariant,
) -> SpatiotemporalWindowDataset:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported temporal split: {split}")
    legacy = variant == "legacy_retrained"
    mask_mode: MaskMode = "point" if legacy else "mixed"
    point_rate = (
        (config.legacy_point_rate, config.legacy_point_rate)
        if legacy
        else (config.point_rate_min, config.point_rate_max)
    )
    return SpatiotemporalWindowDataset(
        panel,
        split,  # type: ignore[arg-type]
        seq_len=config.seq_len,
        stride=config.stride if split == "train" else config.seq_len,
        max_nodes=config.max_nodes,
        min_context_points=config.min_context_points,
        seed=config.seed,
        mask_mode=mask_mode,
        point_rate=point_rate,
        block_lengths=config.block_lengths,
    )


def _loader(
    dataset: SpatiotemporalWindowDataset,
    config: BiLSTMTrainingConfig,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    loader_generator = generator or torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        generator=loader_generator,
    )


def _training_loss(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
    variant: BiLSTMVariant,
    *,
    direction: bool,
) -> torch.Tensor:
    error = normalized_error(prediction, batch["target"], direction=direction)
    if variant == "fair":
        mask = batch["evaluation_mask"].bool()
        if not mask.any():
            raise RuntimeError("Fair BiLSTM batch has no hidden targets")
        return error[mask].abs().mean()
    mask = batch["visible_mask"].bool() | batch["evaluation_mask"].bool()
    if not mask.any():
        raise RuntimeError("Legacy BiLSTM batch has no observed targets")
    return error[mask].square().mean()


def _save_checkpoint(
    path: Path,
    model: MeteorologicalBiLSTM,
    variable: str,
    model_config: BiLSTMConfig,
    training_config: BiLSTMTrainingConfig,
    validation: EvaluationResult,
    training_run: dict[str, Any],
    training_dataset: dict[str, Any],
) -> None:
    atomic_torch_save(
        {
            "format_version": 2,
            "model_class": "MeteorologicalBiLSTM",
            "variable": variable,
            "variant": model_config.variant,
            "state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "training_run": training_run,
            "training_dataset": training_dataset,
            "recipe": _recipe(model_config.variant),
            "validation": asdict(validation),
            "validation_postprocessing": prediction_policy(),
        },
        path,
    )


def _save_training_progress(
    path: Path,
    model: MeteorologicalBiLSTM,
    optimizer: torch.optim.Optimizer,
    variable: str,
    model_config: BiLSTMConfig,
    training_config: BiLSTMTrainingConfig,
    best_state: dict[str, torch.Tensor],
    best_result: EvaluationResult,
    training_state: dict[str, Any],
    rng_state: dict[str, Any],
    training_dataset: dict[str, Any],
) -> None:
    atomic_torch_save(
        {
            "format_version": 1,
            "checkpoint_kind": "training_progress",
            "model_class": "MeteorologicalBiLSTM",
            "variable": variable,
            "variant": model_config.variant,
            "state_dict": model.state_dict(),
            "best_state_dict": best_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "best_validation": asdict(best_result),
            "validation_postprocessing": prediction_policy(),
            "training_state": training_state,
            "rng_state": rng_state,
            "training_dataset": training_dataset,
        },
        path,
    )


def _completed_result(
    path: Path,
    variable: str,
    model_config: BiLSTMConfig,
    config: BiLSTMTrainingConfig,
    device: torch.device,
    training_dataset: dict[str, Any],
) -> EvaluationResult:
    raw = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("Completed checkpoint must contain a mapping")
    if raw.get("model_class") != "MeteorologicalBiLSTM":
        raise ValueError("Completed checkpoint targets a different model")
    if raw.get("variable") != variable or raw.get("variant") != model_config.variant:
        raise ValueError("Completed checkpoint targets a different variable or variant")
    if raw.get("model_config") != asdict(model_config):
        raise ValueError("Completed checkpoint model configuration has changed")
    if raw.get("training_config") != asdict(config):
        raise ValueError("Completed checkpoint training configuration has changed")
    if raw.get("validation_postprocessing") != prediction_policy():
        raise ValueError(
            "Completed checkpoint uses different validation postprocessing; use a new training output directory"
        )
    validate_training_dataset(raw, training_dataset)
    validation = raw.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Completed checkpoint lacks validation metrics")
    return EvaluationResult(**validation)


def load_bilstm_checkpoint(
    path: str | Path,
    device: torch.device | None = None,
) -> tuple[MeteorologicalBiLSTM, dict[str, Any]]:
    """Load a project checkpoint with PyTorch's restricted deserializer."""
    target_device = device or default_device()
    raw = torch.load(Path(path), map_location=target_device, weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("BiLSTM checkpoint must contain a mapping")
    if raw.get("model_class") != "MeteorologicalBiLSTM":
        raise ValueError("Checkpoint is not a MeteorologicalBiLSTM")
    config_raw = raw.get("model_config")
    state_dict = raw.get("state_dict")
    if not isinstance(config_raw, dict) or not isinstance(state_dict, dict):
        raise ValueError("BiLSTM checkpoint lacks model_config or state_dict")
    model = MeteorologicalBiLSTM(BiLSTMConfig(**config_raw))
    model.load_state_dict(state_dict, strict=True)
    model.to(target_device).eval()
    return model, raw


def train_bilstm(  # noqa: PLR0914
    data_root: Path,
    variable: str,
    output_path: Path,
    model_config: BiLSTMConfig,
    config: BiLSTMTrainingConfig,
    device: torch.device,
    *,
    resume: bool = False,
    progress_every: int = 100,
) -> EvaluationResult:
    """Train one variable and one explicitly named BiLSTM recipe."""
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    resume_path = progress_checkpoint_path(output_path)
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"Completed checkpoint exists at {output_path}; use --resume or a new output directory"
        )
    dataset_identity = training_dataset_identity(data_root, variable)
    if resume and output_path.exists() and not resume_path.exists():
        logger.info("Skipping completed training at %s", output_path)
        return _completed_result(
            output_path, variable, model_config, config, device, dataset_identity
        )
    if resume_path.exists() and not resume:
        raise FileExistsError(
            f"Unfinished training exists at {resume_path}; rerun with --resume"
        )
    started = time.perf_counter()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    panel = build_spatiotemporal_panel(data_root, variable)
    train_dataset = _make_dataset(panel, "train", config, model_config.variant)
    validation_dataset = _make_dataset(panel, "val", config, model_config.variant)
    train_generator = torch.Generator().manual_seed(config.seed)
    train_loader = _loader(
        train_dataset,
        config,
        shuffle=True,
        generator=train_generator,
    )
    validation_loader = _loader(validation_dataset, config, shuffle=False)
    model = MeteorologicalBiLSTM(model_config).to(device)
    learning_rate = (
        config.legacy_learning_rate
        if model_config.variant == "legacy_retrained"
        else config.learning_rate
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
    )
    direction = variable == "wind_direction"
    start_epoch = 0
    elapsed_before_resume = 0.0
    resume_count = 0
    history: list[dict[str, Any]]
    stopped_early = False
    if resume and resume_path.exists():
        progress = load_training_progress(
            resume_path,
            device,
            model_class="MeteorologicalBiLSTM",
            variable=variable,
            variant=model_config.variant,
            model_config=asdict(model_config),
            training_config=asdict(config),
            training_dataset=dataset_identity,
        )
        model.load_state_dict(progress["state_dict"], strict=True)
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        best_state = progress["best_state_dict"]
        best_result = EvaluationResult(**progress["best_validation"])
        training_state = progress["training_state"]
        start_epoch = int(training_state["epochs_completed"])
        best_epoch = int(training_state["best_epoch"])
        stale_epochs = int(training_state["stale_epochs"])
        optimizer_steps = int(training_state["optimizer_steps"])
        history = list(training_state["history"])
        elapsed_before_resume = float(training_state["elapsed_seconds"])
        resume_count = int(training_state.get("resume_count", 0)) + 1
        restore_rng_state(progress["rng_state"], device, train_generator)
        stopped_early = stale_epochs >= config.patience
        logger.info(
            "Resuming %s/%s at epoch %d/%d from %s",
            variable,
            model_config.variant,
            start_epoch + 1,
            config.max_epochs,
            resume_path,
        )
    else:
        best_result = evaluate_model(
            model,
            validation_loader,
            device,
            variable,
            max_batches=config.max_validation_batches,
        )
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        stale_epochs = 0
        optimizer_steps = 0
        history = []

    epoch_range = () if stopped_early else range(start_epoch, config.max_epochs)
    for epoch in epoch_range:
        train_dataset.set_epoch(epoch)
        model.train()
        epoch_started = time.perf_counter()
        total_loss = 0.0
        batches = 0
        total_batches = (
            len(train_loader)
            if config.max_batches_per_epoch is None
            else min(len(train_loader), config.max_batches_per_epoch)
        )
        for batch_index, cpu_batch in enumerate(train_loader):
            if (
                config.max_batches_per_epoch is not None
                and batch_index >= config.max_batches_per_epoch
            ):
                break
            batch = to_device(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _residual = model(
                batch["features"],
                batch["coordinates"],
                batch["node_mask"],
                batch["baseline"],
            )
            loss = _training_loss(
                prediction,
                batch,
                model_config.variant,
                direction=direction,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
            optimizer_steps += 1
            if progress_every and (
                batches % progress_every == 0 or batches == total_batches
            ):
                epoch_elapsed = time.perf_counter() - epoch_started
                logger.info(
                    "%s/%s epoch %d: batch %d/%d (%.1f%%), loss %.6f, %.1f batch/s",
                    variable,
                    model_config.variant,
                    epoch + 1,
                    batches,
                    total_batches,
                    100.0 * batches / max(total_batches, 1),
                    total_loss / batches,
                    batches / max(epoch_elapsed, 1e-9),
                )
        result = evaluate_model(
            model,
            validation_loader,
            device,
            variable,
            max_batches=config.max_validation_batches,
        )
        logger.info(
            "%s/%s epoch %d: loss %.6f, val MAE %.6f, interpolation %.6f",
            variable,
            model_config.variant,
            epoch + 1,
            total_loss / max(batches, 1),
            result.model_mae,
            result.baseline_mae,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(batches, 1),
                "train_batches": batches,
                "optimizer_steps": optimizer_steps,
                "validation": asdict(result),
            }
        )
        if result.model_mae < best_result.model_mae - 1e-6:
            best_result = result
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
        elapsed_seconds = elapsed_before_resume + time.perf_counter() - started
        _save_training_progress(
            resume_path,
            model,
            optimizer,
            variable,
            model_config,
            config,
            best_state,
            best_result,
            {
                "best_epoch": best_epoch,
                "epochs_completed": len(history),
                "stale_epochs": stale_epochs,
                "optimizer_steps": optimizer_steps,
                "elapsed_seconds": elapsed_seconds,
                "resume_count": resume_count,
                "history": history,
            },
            capture_rng_state(device, train_generator),
            dataset_identity,
        )
        logger.info(
            "Saved resumable progress through epoch %d to %s",
            epoch + 1,
            resume_path,
        )
        if stale_epochs >= config.patience:
            stopped_early = True
            break
    model.load_state_dict(best_state, strict=True)
    _save_checkpoint(
        output_path,
        model,
        variable,
        model_config,
        config,
        best_result,
        {
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "optimizer_steps": optimizer_steps,
            "available_train_batches_per_epoch": len(train_loader),
            "used_all_train_batches_each_epoch": all(
                item["train_batches"] == len(train_loader) for item in history
            ),
            "stopped_early": stopped_early,
            "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started,
            "resume_count": resume_count,
            "history": history,
        },
        dataset_identity,
    )
    resume_path.unlink(missing_ok=True)
    return best_result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse baseline training arguments."""
    parser = argparse.ArgumentParser(
        description="Train legacy-retrained and fair BiLSTM baselines"
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/2-processed-v2"))
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=VARIABLE_TYPE_NAMES,
        default=("temperature",),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("legacy_retrained", "fair"),
        default=("legacy_retrained", "fair"),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/bilstm")
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="train one independent checkpoint per seed",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume unfinished runs and skip compatible completed checkpoints",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        metavar="BATCHES",
        help="log progress every N batches; use 0 to disable (default: 100)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train requested variables and variants sequentially."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args(argv)
    device = default_device()
    summaries: dict[str, dict[str, Any]] = {}
    for variant_raw in args.variants:
        variant: BiLSTMVariant = variant_raw
        model_config, training_config = read_bilstm_config(args.config, variant)
        if args.quick:
            training_config = replace(
                training_config,
                max_epochs=2,
                max_batches_per_epoch=8,
                max_validation_batches=8,
                patience=2,
            )
        seeds = experiment_seeds(training_config.seed, args.seeds)
        seed_specific_paths = args.seeds is not None
        for seed in seeds:
            seeded_config = replace(training_config, seed=seed)
            for variable in args.variables:
                output_path = bilstm_checkpoint_path(
                    args.checkpoint_dir,
                    variant,
                    variable,
                    seed=seed if seed_specific_paths else None,
                )
                try:
                    result = train_bilstm(
                        args.data_dir,
                        variable,
                        output_path,
                        model_config,
                        seeded_config,
                        device,
                        resume=args.resume,
                        progress_every=args.progress_every,
                    )
                except KeyboardInterrupt:
                    logger.warning(
                        "Training interrupted; rerun the same command with --resume. "
                        "Progress through the last completed epoch is safe."
                    )
                    return 130
                summary_key = (
                    f"{variant}/seed-{seed}/{variable}"
                    if seed_specific_paths
                    else f"{variant}/{variable}"
                )
                summaries[summary_key] = asdict(result)
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
