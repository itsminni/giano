"""Create non-destructive imputed NetCDF products from natural gaps."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from giano.meteorology import PHYSICAL_BOUNDS_BY_VARIABLE
from giano.model.train_imputeformer import (
    TrainingConfig,
    load_imputeformer_checkpoint,
)
from giano.netcdf import open_dataset_robust
from giano.prediction import physical_prediction, prediction_policy
from giano.provenance import file_sha256
from giano.runtime import default_device
from giano.spatiotemporal_dataset import (
    NaturalGapWindowDataset,
    SpatiotemporalWindowDataset,
    _temporal_baseline,
    build_spatiotemporal_panel,
)
from giano.variables import VARIABLE_TYPE_NAMES

logger = logging.getLogger(__name__)
_NS_PER_HOUR = 3_600_000_000_000
MODEL_FAMILIES = ("imputeformer", "bilstm_fair", "interpolation")


class _InterpolationBaseline(torch.nn.Module):
    def forward(self, features, coordinates, node_mask, baseline):
        return baseline, torch.zeros_like(baseline)


def _checkpoint_training_config(metadata: dict[str, Any]) -> TrainingConfig:
    raw = metadata.get("training_config")
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint lacks training_config")
    # Read shared window settings from either family's checkpoint.
    values = {
        key: value
        for key, value in raw.items()
        if key in TrainingConfig.__dataclass_fields__
    }
    values["block_lengths"] = tuple(values.get("block_lengths", (3, 6, 12, 24)))
    config = TrainingConfig(**values)
    config.validate()
    return config


@torch.no_grad()
def impute_variable(  # noqa: PLR0914, PLR0915
    data_root: Path,
    checkpoint_path: Path | None,
    output_dir: Path,
    *,
    batch_size: int = 4,
    device: torch.device | None = None,
    model_family: str = "imputeformer",
    variable: str | None = None,
    fallback: str = "interpolation",
) -> list[Path]:
    """Impute internal natural gaps while preserving original observations."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported inference family: {model_family}")
    if fallback not in {"interpolation", "none"}:
        raise ValueError(f"Unsupported uncovered-gap fallback: {fallback}")
    target_device = device or default_device()
    requested_variable = variable
    model: torch.nn.Module
    if model_family == "interpolation":
        if checkpoint_path is not None:
            raise ValueError("Interpolation does not use a checkpoint")
        model = _InterpolationBaseline().to(target_device)
        config = TrainingConfig()
    else:
        if checkpoint_path is None:
            raise ValueError("Neural inference requires a checkpoint")
        if model_family == "imputeformer":
            model, metadata = load_imputeformer_checkpoint(
                checkpoint_path, target_device
            )
        else:
            # Load the optional comparison family only when explicitly selected.
            from giano.baselines.train_bilstm import load_bilstm_checkpoint

            model, metadata = load_bilstm_checkpoint(checkpoint_path, target_device)
            if metadata.get("variant") != "fair":
                raise ValueError(
                    "Natural-gap inference supports only the fair BiLSTM variant"
                )
        variable = metadata.get("variable")
        if requested_variable is not None and variable != requested_variable:
            raise ValueError("Requested variable does not match the checkpoint")
        config = _checkpoint_training_config(metadata)
    if not isinstance(variable, str) or variable not in VARIABLE_TYPE_NAMES:
        raise ValueError("Inference requires a valid meteorological variable")
    panel = build_spatiotemporal_panel(data_root, variable)
    dataset = SpatiotemporalWindowDataset(
        panel,
        "all",
        seq_len=config.seq_len,
        stride=max(1, config.seq_len // 2),
        max_nodes=config.max_nodes,
        min_context_points=config.min_context_points,
        seed=config.seed,
        mask_mode="natural",
        point_rate=(config.point_rate_min, config.point_rate_max),
        block_lengths=config.block_lengths,
    )
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        NaturalGapWindowDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    shape = panel.values.shape
    prediction_count = np.zeros(shape, dtype=np.uint16)
    has_learned_correction = np.zeros(shape, dtype=bool)
    direction = variable == "wind_direction"
    prediction_sum: np.ndarray | None
    prediction_sin: np.ndarray | None
    prediction_cos: np.ndarray | None
    if direction:
        prediction_sin = np.zeros(shape, dtype=np.float32)
        prediction_cos = np.zeros(shape, dtype=np.float32)
        prediction_sum = None
    else:
        prediction_sum = np.zeros(shape, dtype=np.float32)
        prediction_sin = prediction_cos = None

    model.eval()
    logger.info(
        "%s: %d natural-window batches on %s", variable, len(loader), target_device
    )
    for batch_index, cpu_batch in enumerate(loader, start=1):
        features = cpu_batch["features"].to(target_device)
        coordinates = cpu_batch["coordinates"].to(target_device)
        node_mask = cpu_batch["node_mask"].to(target_device)
        baseline = cpu_batch["baseline"].to(target_device)
        normalized, residual = model(features, coordinates, node_mask, baseline)
        physical = (
            physical_prediction(
                normalized,
                cpu_batch["center"].to(target_device),
                cpu_batch["scale"].to(target_device),
                variable=variable,
            )
            .cpu()
            .numpy()
        )
        residual_np = residual.cpu().numpy()
        active_masks = cpu_batch["evaluation_mask"].numpy().astype(bool)
        node_indices = cpu_batch["node_indices"].numpy()
        starts = cpu_batch["start_index"].numpy()

        for row, start in enumerate(starts):
            for local_node, global_node in enumerate(node_indices[row]):
                if global_node < 0:
                    continue
                active = active_masks[row, :, local_node]
                if not active.any():
                    continue
                time_indices = int(start) + np.flatnonzero(active)
                predictions = physical[row, active, local_node]
                if direction:
                    if prediction_sin is None or prediction_cos is None:
                        raise RuntimeError("Circular accumulators were not initialized")
                    radians = np.deg2rad(predictions)
                    prediction_sin[time_indices, global_node] += np.sin(radians)
                    prediction_cos[time_indices, global_node] += np.cos(radians)
                else:
                    if prediction_sum is None:
                        raise RuntimeError("Scalar accumulator was not initialized")
                    prediction_sum[time_indices, global_node] += predictions
                prediction_count[time_indices, global_node] += 1
                learned = np.abs(residual_np[row, active, local_node]) > 1e-7
                has_learned_correction[time_indices[learned], global_node] = True
        if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
            logger.info(
                "%s: natural inference batch %d/%d", variable, batch_index, len(loader)
            )

    averaged = np.full(shape, np.nan, dtype=np.float32)
    available = prediction_count > 0
    if direction:
        if prediction_sin is None or prediction_cos is None:
            raise RuntimeError("Circular accumulators were not initialized")
        averaged[available] = np.mod(
            np.rad2deg(
                np.arctan2(
                    prediction_sin[available] / prediction_count[available],
                    prediction_cos[available] / prediction_count[available],
                ),
            ),
            360.0,
        )
    else:
        if prediction_sum is None:
            raise RuntimeError("Scalar accumulator was not initialized")
        averaged[available] = prediction_sum[available] / prediction_count[available]

    fallback_panel = None
    if fallback == "interpolation":
        fallback_panel = _temporal_baseline(panel.values, variable)
        if variable == "wind_direction":
            fallback_panel = np.mod(fallback_panel, 360.0)
        else:
            fallback_panel = np.clip(
                fallback_panel, *PHYSICAL_BOUNDS_BY_VARIABLE[variable]
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    panel_start_ns = int(panel.times[0].astype(np.int64))
    for node, source_path in enumerate(panel.source_paths):
        with open_dataset_robust(source_path) as source:
            output = source.load()
        times_ns = (
            np.asarray(output["time"].values).astype("datetime64[ns]").astype(np.int64)
        )
        offsets, remainders = np.divmod(times_ns - panel_start_ns, _NS_PER_HOUR)
        if np.any(remainders != 0):
            raise ValueError(f"Non-hourly source timestamps: {source_path}")
        original = np.asarray(output["value"].values, dtype=np.float64)
        missing = ~np.isfinite(original)
        predictions = averaged[offsets, node]
        filled = original.copy()
        model_available = missing & np.isfinite(predictions)
        filled[model_available] = predictions[model_available]
        if fallback_panel is not None:
            fallback_values = fallback_panel[offsets, node]
            fallback_available = (
                missing & ~model_available & np.isfinite(fallback_values)
            )
            filled[fallback_available] = fallback_values[fallback_available]
        imputed = missing & np.isfinite(filled)
        method = np.zeros(len(filled), dtype=np.int8)
        method[imputed] = 1
        learned = has_learned_correction[offsets, node]
        if model_family == "imputeformer":
            method[imputed & learned] = 2
        elif model_family == "bilstm_fair":
            method[model_available] = 3

        output["imputed_value"] = xr.DataArray(filled, dims=output["value"].dims)
        output["window_prediction_count"] = xr.DataArray(
            prediction_count[offsets, node],
            dims=output["value"].dims,
            attrs={
                "description": "Number of natural input windows supporting the estimate"
            },
        )
        output["imputed_mask"] = xr.DataArray(
            imputed.astype(np.int8),
            dims=output["value"].dims,
            attrs={"codes": "0=original_or_unfilled, 1=imputed"},
        )
        output["imputation_method"] = xr.DataArray(
            method,
            dims=output["value"].dims,
            attrs={
                "codes": "0=original_or_unfilled, 1=interpolation, 2=imputeformer, 3=bilstm_fair"
            },
        )
        output["imputed_value"].attrs.update(output["value"].attrs)
        output["imputed_value"].attrs.update(
            {
                "source_variable": "value",
                "imputation_model": {
                    "imputeformer": "Giano coordinate-inductive ImputeFormer",
                    "bilstm_fair": "Meteorological BiLSTM fair",
                    "interpolation": "Temporal interpolation",
                }[model_family],
                "imputed_points": int(imputed.sum()),
            },
        )
        destination = output_dir / f"{source_path.stem}_imputed.nc"
        output.to_netcdf(destination, engine="h5netcdf")
        output_paths.append(destination)

    manifest = {
        "schema_version": 1,
        "variable": variable,
        "model_family": model_family,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_sha256": file_sha256(checkpoint_path)
        if checkpoint_path is not None
        else None,
        "source_files": len(panel.source_paths),
        "output_files": [path.name for path in output_paths],
        "prediction_postprocessing": prediction_policy(),
        "neural_input_contract": "visible-window-v2"
        if model_family != "interpolation"
        else None,
        "neural_max_nodes": config.max_nodes
        if model_family != "interpolation"
        else None,
        "window_input_contract": "visible-window-v2",
        "window_length": config.seq_len,
        "window_max_nodes": config.max_nodes,
        "fallback": "full_history_interpolation_for_uncovered_gaps"
        if fallback == "interpolation"
        else "none",
    }
    (output_dir / f"{variable}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Write natural-gap reconstructions with an explicitly chosen method",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="checkpoint directory, required for Giano and BiLSTM",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/imputed"),
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=VARIABLE_TYPE_NAMES,
        default=VARIABLE_TYPE_NAMES,
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--model-family", choices=MODEL_FAMILIES, default="imputeformer"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="load seed-N/variable.pt below the checkpoint directory",
    )
    parser.add_argument(
        "--fallback",
        choices=("interpolation", "none"),
        default="interpolation",
        help="Use none to leave gaps without local model context unfilled",
    )
    args = parser.parse_args(argv)
    if args.model_family == "interpolation":
        if args.checkpoint_dir is not None or args.seed is not None:
            parser.error("Interpolation does not use checkpoints or a training seed")
    elif args.checkpoint_dir is None:
        parser.error("--checkpoint-dir is required for neural inference")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("seed must be inside [0, 2**32)")
    return args


def main(argv: list[str] | None = None) -> int:
    """Impute each selected variable sequentially to bound memory usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args(argv)
    directory = args.checkpoint_dir
    if args.seed is not None:
        directory = directory / f"seed-{args.seed}"
    for variable in args.variables:
        checkpoint = directory / f"{variable}.pt" if directory is not None else None
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        paths = impute_variable(
            args.data_dir,
            checkpoint,
            args.output_dir / variable,
            batch_size=args.batch_size,
            model_family=args.model_family,
            variable=variable,
            fallback=args.fallback,
        )
        logger.info("%s: wrote %d reconstructed files", variable, len(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
