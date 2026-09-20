"""Crash-safe, restricted checkpoints for resumable model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from giano.prediction import prediction_policy
from giano.provenance import validate_training_dataset


def progress_checkpoint_path(output_path: Path) -> Path:
    """Return the sidecar used to resume an unfinished training run."""
    return output_path.with_name(f"{output_path.stem}.resume{output_path.suffix}")


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write one PyTorch artifact without exposing a partial destination file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def capture_rng_state(
    device: torch.device,
    train_generator: torch.Generator,
) -> dict[str, Any]:
    """Capture stochastic state needed at the next epoch boundary."""
    state: dict[str, Any] = {
        "cpu": torch.get_rng_state(),
        "train_loader": train_generator.get_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(
    state: dict[str, Any],
    device: torch.device,
    train_generator: torch.Generator,
) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    cpu_state = state.get("cpu")
    loader_state = state.get("train_loader")
    if not isinstance(cpu_state, torch.Tensor) or not isinstance(
        loader_state, torch.Tensor
    ):
        raise ValueError("Training checkpoint lacks CPU or loader RNG state")
    torch.set_rng_state(cpu_state.cpu())
    train_generator.set_state(loader_state.cpu())
    if device.type == "cuda" and "cuda" in state:
        cuda_states = state["cuda"]
        if not isinstance(cuda_states, list) or not all(
            isinstance(item, torch.Tensor) for item in cuda_states
        ):
            raise ValueError("Training checkpoint has invalid CUDA RNG state")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    elif device.type == "mps" and "mps" in state:
        mps_state = state["mps"]
        if not isinstance(mps_state, torch.Tensor):
            raise ValueError("Training checkpoint has invalid MPS RNG state")
        torch.mps.set_rng_state(mps_state.cpu())


def load_training_progress(
    path: Path,
    device: torch.device,
    *,
    model_class: str,
    variable: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    training_dataset: dict[str, Any],
    variant: str | None = None,
) -> dict[str, Any]:
    """Load and validate a project-generated resumable checkpoint."""
    raw = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("Training checkpoint must contain a mapping")
    if raw.get("checkpoint_kind") != "training_progress":
        raise ValueError("Checkpoint is not resumable training progress")
    if raw.get("model_class") != model_class or raw.get("variable") != variable:
        raise ValueError("Training checkpoint targets a different model or variable")
    if variant is not None and raw.get("variant") != variant:
        raise ValueError("Training checkpoint targets a different model variant")
    if raw.get("model_config") != model_config:
        raise ValueError("Training checkpoint model configuration has changed")
    if raw.get("training_config") != training_config:
        raise ValueError("Training checkpoint training configuration has changed")
    if raw.get("validation_postprocessing") != prediction_policy():
        raise ValueError(
            "Training progress uses different validation postprocessing; retain this checkpoint and use a new output directory"
        )
    validate_training_dataset(raw, training_dataset)
    return raw
