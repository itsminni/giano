"""Train and benchmark the meteorological ImputeFormer adaptation."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from giano.evaluation.gapfill.results import (
    EvaluationResult,
    evaluate_model,
    normalized_error,
    to_device,
)
from giano.experiments import experiment_seeds, imputeformer_checkpoint_path
from giano.model.imputeformer import (
    ImputeFormerConfig,
    MeteorologicalImputeFormer,
    temporal_spectral_regularization,
)
from giano.prediction import prediction_policy
from giano.provenance import training_dataset_identity, validate_training_dataset
from giano.runtime import default_device
from giano.spatiotemporal_dataset import (
    INPUT_FEATURES,
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
class TrainingConfig:
    """Training and masking settings for one variable model."""

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
    weight_decay: float = 1e-4
    patience: int = 4
    gradient_clip: float = 1.0
    # This key is recorded in the released checkpoint configuration.
    low_rank_weight: float = 1e-3
    # Recorded in released checkpoints; unused because observed residuals are zero.
    context_weight: float = 0.0
    residual_weight: float = 1e-3
    seed: int = 42
    point_rate_min: float = 0.15
    point_rate_max: float = 0.45
    block_lengths: tuple[int, ...] = (3, 6, 12, 24)
    train_mask_mode: MaskMode = "mixed"
    validation_mask_mode: MaskMode = "mixed"

    @classmethod
    def from_checkpoint(cls, metadata: dict[str, Any]) -> TrainingConfig:
        """Read and validate the configuration saved with a model."""
        raw = metadata.get("training_config")
        if not isinstance(raw, dict):
            raise ValueError("Checkpoint lacks training_config")
        values = dict(raw)
        if "block_lengths" in values:
            values["block_lengths"] = tuple(values["block_lengths"])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        """Validate values before allocating a panel."""
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
            raise ValueError("Training dimensions and limits must be positive")
        if self.max_batches_per_epoch is not None and self.max_batches_per_epoch <= 0:
            raise ValueError("max_batches_per_epoch must be positive or null")
        if not 0 < self.point_rate_min <= self.point_rate_max < 1:
            raise ValueError("Invalid point mask rates")
        weights = (self.low_rank_weight, self.context_weight, self.residual_weight)
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("Loss weights must be finite and non-negative")


def training_objective() -> dict[str, str]:
    """Describe the loss semantics independently of saved numeric weights."""
    return {
        "version": "masked-temporal-spectrum-v1",
        "reconstruction": "synthetic-hidden-normalized-mae",
        "residual_penalty": "synthetic-hidden-l1",
        "spectral_completion": "visible-target-otherwise-prediction",
        "spectral_reduction": "time-only-valid-station-mean",
        "direction_representation": "joint-sin-cos-spectrum",
        "context_penalty": "disabled-observations-hard-gated",
    }


def _validate_training_objective(raw: dict[str, Any]) -> None:
    if raw.get("training_objective") != training_objective():
        raise ValueError(
            "Checkpoint training objective is missing or different; retain the old "
            "checkpoint for inference and use a new training output directory"
        )


def _read_config(path: Path) -> tuple[ImputeFormerConfig, TrainingConfig]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration top-level must be a mapping")
    section = raw.get("imputeformer", {})
    if not isinstance(section, dict):
        raise ValueError("imputeformer configuration must be a mapping")
    model_raw = section.get("model", {})
    training_raw = section.get("training", {})
    if not isinstance(model_raw, dict) or not isinstance(training_raw, dict):
        raise ValueError("imputeformer model/training sections must be mappings")
    model_values = {
        key: value
        for key, value in model_raw.items()
        if key in ImputeFormerConfig.__dataclass_fields__
    }
    training_values = {
        key: value
        for key, value in training_raw.items()
        if key in TrainingConfig.__dataclass_fields__
    }
    if "block_lengths" in training_values:
        training_values["block_lengths"] = tuple(training_values["block_lengths"])
    model_config = ImputeFormerConfig(**model_values)
    training_config = TrainingConfig(**training_values)
    model_config.validate()
    training_config.validate()
    if model_config.input_dim != len(INPUT_FEATURES):
        raise ValueError(
            f"model.input_dim must equal the {len(INPUT_FEATURES)} dataset features",
        )
    return model_config, training_config


def _make_dataset(
    panel: SpatiotemporalPanel,
    split: str,
    config: TrainingConfig,
    *,
    mask_mode: MaskMode,
) -> SpatiotemporalWindowDataset:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported temporal split: {split}")
    stride = config.stride if split == "train" else config.seq_len
    return SpatiotemporalWindowDataset(
        panel,
        split,  # type: ignore[arg-type]
        seq_len=config.seq_len,
        stride=stride,
        max_nodes=config.max_nodes,
        min_context_points=config.min_context_points,
        seed=config.seed,
        mask_mode=mask_mode,
        point_rate=(config.point_rate_min, config.point_rate_max),
        block_lengths=config.block_lengths,
    )


def _loader(
    dataset: SpatiotemporalWindowDataset,
    config: TrainingConfig,
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


def _loss(
    batch: dict[str, torch.Tensor],
    prediction: torch.Tensor,
    residual: torch.Tensor,
    config: TrainingConfig,
    *,
    direction: bool,
) -> torch.Tensor:
    evaluation_mask = batch["evaluation_mask"].bool()
    error = normalized_error(
        prediction,
        batch["target"],
        direction=direction,
    )
    if not evaluation_mask.any():
        raise RuntimeError("Training batch has no synthetically hidden targets")
    reconstruction = error[evaluation_mask].abs().mean()
    residual_penalty = residual[evaluation_mask].abs().mean()
    loss = reconstruction + config.residual_weight * residual_penalty
    if config.low_rank_weight:
        frequency = temporal_spectral_regularization(
            prediction,
            batch["target"],
            visible_mask=batch["visible_mask"],
            node_mask=batch["node_mask"],
            direction=direction,
        )
        loss = loss + config.low_rank_weight * frequency
    return loss


def _save_checkpoint(
    path: Path,
    model: MeteorologicalImputeFormer,
    variable: str,
    model_config: ImputeFormerConfig,
    training_config: TrainingConfig,
    validation: EvaluationResult,
    training_run: dict[str, Any],
    training_dataset: dict[str, Any],
) -> None:
    atomic_torch_save(
        {
            "format_version": 2,
            "model_class": "MeteorologicalImputeFormer",
            "variable": variable,
            "state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "training_objective": training_objective(),
            "training_run": training_run,
            "training_dataset": training_dataset,
            "validation": asdict(validation),
            "validation_postprocessing": prediction_policy(),
        },
        path,
    )


def _save_training_progress(
    path: Path,
    model: MeteorologicalImputeFormer,
    optimizer: torch.optim.Optimizer,
    variable: str,
    model_config: ImputeFormerConfig,
    training_config: TrainingConfig,
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
            "model_class": "MeteorologicalImputeFormer",
            "variable": variable,
            "state_dict": model.state_dict(),
            "best_state_dict": best_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "training_objective": training_objective(),
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
    model_config: ImputeFormerConfig,
    config: TrainingConfig,
    device: torch.device,
    training_dataset: dict[str, Any],
) -> EvaluationResult:
    raw = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("Completed checkpoint must contain a mapping")
    if raw.get("model_class") != "MeteorologicalImputeFormer":
        raise ValueError("Completed checkpoint targets a different model")
    if raw.get("variable") != variable:
        raise ValueError("Completed checkpoint targets a different variable")
    if raw.get("model_config") != asdict(model_config):
        raise ValueError("Completed checkpoint model configuration has changed")
    _validate_training_objective(raw)
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


def load_imputeformer_checkpoint(
    path: str | Path,
    device: torch.device | None = None,
) -> tuple[MeteorologicalImputeFormer, dict[str, Any]]:
    """Load a project-generated checkpoint through PyTorch's restricted loader."""
    target_device = device or default_device()
    raw = torch.load(Path(path), map_location=target_device, weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("ImputeFormer checkpoint must contain a mapping")
    if raw.get("model_class") != "MeteorologicalImputeFormer":
        raise ValueError("Checkpoint is not a MeteorologicalImputeFormer")
    config_raw = raw.get("model_config")
    state_dict = raw.get("state_dict")
    if not isinstance(config_raw, dict) or not isinstance(state_dict, dict):
        raise ValueError("Checkpoint lacks model_config or state_dict")
    model = MeteorologicalImputeFormer(ImputeFormerConfig(**config_raw))
    model.load_state_dict(state_dict, strict=True)
    model.to(target_device).eval()
    return model, raw


def train_variable(  # noqa: PLR0914
    data_root: Path,
    variable: str,
    output_path: Path,
    model_config: ImputeFormerConfig,
    config: TrainingConfig,
    device: torch.device,
    *,
    resume: bool = False,
    progress_every: int = 100,
) -> EvaluationResult:
    """Train one variable-specific set of ImputeFormer weights."""
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    model_config.validate()
    config.validate()
    if config.context_weight:
        logger.warning(
            "context_weight is deprecated and ignored: observed residuals are "
            "already zero by construction; remove this key from new configurations"
        )
    resume_path = progress_checkpoint_path(output_path)
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"Completed checkpoint exists at {output_path}; use --resume or a new output directory"
        )
    dataset_identity = training_dataset_identity(data_root, variable)
    if resume and output_path.exists() and not resume_path.exists():
        completed_result = _completed_result(
            output_path, variable, model_config, config, device, dataset_identity
        )
        logger.info("Skipping compatible completed training at %s", output_path)
        return completed_result
    if resume_path.exists() and not resume:
        raise FileExistsError(
            f"Unfinished training exists at {resume_path}; rerun with --resume"
        )
    started = time.perf_counter()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    panel = build_spatiotemporal_panel(data_root, variable)
    train_dataset = _make_dataset(
        panel,
        "train",
        config,
        mask_mode=config.train_mask_mode,
    )
    validation_dataset = _make_dataset(
        panel,
        "val",
        config,
        mask_mode=config.validation_mask_mode,
    )
    train_generator = torch.Generator().manual_seed(config.seed)
    train_loader = _loader(
        train_dataset,
        config,
        shuffle=True,
        generator=train_generator,
    )
    validation_loader = _loader(validation_dataset, config, shuffle=False)
    model = MeteorologicalImputeFormer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
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
            model_class="MeteorologicalImputeFormer",
            variable=variable,
            model_config=asdict(model_config),
            training_config=asdict(config),
            training_dataset=dataset_identity,
        )
        _validate_training_objective(progress)
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
            "Resuming %s at epoch %d/%d from %s",
            variable,
            start_epoch + 1,
            config.max_epochs,
            resume_path,
        )
    else:
        baseline_state = copy.deepcopy(model.state_dict())
        best_result = evaluate_model(
            model,
            validation_loader,
            device,
            variable,
            max_batches=config.max_validation_batches,
        )
        best_state = baseline_state
        best_epoch = 0
        stale_epochs = 0
        optimizer_steps = 0
        history = []
        logger.info(
            "%s initial validation: model MAE %.6f, interpolation MAE %.6f",
            variable,
            best_result.model_mae,
            best_result.baseline_mae,
        )

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
            prediction, residual = model(
                batch["features"],
                batch["coordinates"],
                batch["node_mask"],
                batch["baseline"],
            )
            loss = _loss(
                batch,
                prediction,
                residual,
                config,
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
                    "%s epoch %d: batch %d/%d (%.1f%%), loss %.6f, %.1f batch/s",
                    variable,
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
            "%s epoch %d: loss %.6f, val MAE %.6f, interpolation %.6f, variation %.3f",
            variable,
            epoch + 1,
            total_loss / max(batches, 1),
            result.model_mae,
            result.baseline_mae,
            result.variation_ratio,
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the coordinate-inductive meteorological ImputeFormer",
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/2-processed-v2"))
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=VARIABLE_TYPE_NAMES,
        default=VARIABLE_TYPE_NAMES,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/imputeformer"),
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
    """Train selected variables sequentially to keep panel memory bounded."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args(argv)
    model_config, training_config = _read_config(args.config)
    if args.quick:
        values = asdict(training_config)
        values.update(
            max_epochs=2,
            max_batches_per_epoch=8,
            max_validation_batches=8,
            patience=2,
        )
        training_config = TrainingConfig(**values)
    seeds = experiment_seeds(training_config.seed, args.seeds)
    seed_specific_paths = args.seeds is not None
    device = default_device()
    logger.info("Using device %s", device)
    summaries: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        seeded_config = replace(training_config, seed=seed)
        for variable in args.variables:
            output_path = imputeformer_checkpoint_path(
                args.checkpoint_dir,
                variable,
                seed=seed if seed_specific_paths else None,
            )
            try:
                result = train_variable(
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
            summary_key = f"seed-{seed}/{variable}" if seed_specific_paths else variable
            summaries[summary_key] = asdict(result)
            logger.info("Checkpoint ready at %s", output_path)
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
