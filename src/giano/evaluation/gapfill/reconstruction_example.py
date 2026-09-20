"""Export one inspectable same-mask reconstruction for notebooks and demos."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from giano.evaluation.gapfill.cases import BenchmarkCase, dataset_for_case
from giano.evaluation.gapfill.masks import SyntheticMaskType
from giano.evaluation.gapfill.results import physical_error
from giano.experiments import imputeformer_checkpoint_path
from giano.model.train_imputeformer import TrainingConfig, load_imputeformer_checkpoint
from giano.prediction import physical_prediction, prediction_policy
from giano.provenance import git_provenance
from giano.runtime import default_device
from giano.spatiotemporal_dataset import SpatiotemporalPanel, build_spatiotemporal_panel
from giano.variables import VARIABLE_TYPE_NAMES

DEFAULT_OUTPUT = Path("artifacts/examples/reconstruction_example.json")


def _nullable(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


@torch.no_grad()
def reconstruction_example(  # noqa: PLR0914
    model: torch.nn.Module,
    metadata: dict[str, Any],
    panel: SpatiotemporalPanel,
    *,
    split: str,
    case: BenchmarkCase,
    seed: int,
    sample_index: int | None,
    random_seed: int | None = None,
    device: torch.device,
) -> dict[str, Any]:
    """Return one model reconstruction with its exact target and mask."""
    if split not in {"val", "test"}:
        raise ValueError("reconstruction example split must be val or test")
    config = TrainingConfig.from_checkpoint(metadata)
    dataset = dataset_for_case(panel, split, config, case, seed)
    if sample_index is None:
        sample_index = int(np.random.default_rng(random_seed).integers(len(dataset)))
    if not 0 <= sample_index < len(dataset):
        raise IndexError(
            f"sample_index {sample_index} is outside [0, {len(dataset) - 1}]"
        )
    sample = dataset[sample_index]
    prediction_norm, _residual = model(
        sample["features"].unsqueeze(0).to(device),
        sample["coordinates"].unsqueeze(0).to(device),
        sample["node_mask"].unsqueeze(0).to(device),
        sample["baseline"].unsqueeze(0).to(device),
    )
    prediction = physical_prediction(
        prediction_norm,
        sample["center"].unsqueeze(0).to(device),
        sample["scale"].unsqueeze(0).to(device),
        variable=panel.variable,
    )[0].cpu()
    raw_prediction = physical_prediction(
        prediction_norm,
        sample["center"].unsqueeze(0).to(device),
        sample["scale"].unsqueeze(0).to(device),
        variable=panel.variable,
        constrain=False,
    )[0].cpu()
    hidden_counts = sample["evaluation_mask"].sum(dim=0)
    local_node = int(torch.argmax(hidden_counts))
    if int(hidden_counts[local_node]) == 0:
        raise RuntimeError("selected reconstruction window contains no hidden values")
    global_node = int(sample["node_indices"][local_node])
    if not 0 <= global_node < panel.num_nodes:
        raise RuntimeError("reconstruction example selected a padded node")

    target = sample["target_physical"][:, local_node]
    baseline = sample["baseline_physical"][:, local_node]
    predicted = prediction[:, local_node]
    hidden = sample["evaluation_mask"][:, local_node].bool()
    visible = sample["visible_mask"][:, local_node].bool()
    originally_observed = visible | hidden
    target_values = target.numpy().astype(np.float64)
    baseline_values = baseline.numpy().astype(np.float64)
    prediction_values = predicted.numpy().astype(np.float64)
    hidden_values = hidden.numpy()
    visible_values = visible.numpy()
    observed_values = originally_observed.numpy()
    ground_truth = np.where(observed_values, target_values, np.nan)
    corrupted = np.where(visible_values, target_values, np.nan)
    interpolation = np.where(
        visible_values,
        target_values,
        np.where(hidden_values, baseline_values, np.nan),
    )
    reconstruction = np.where(
        visible_values,
        target_values,
        np.where(hidden_values, prediction_values, np.nan),
    )
    model_error = physical_error(
        predicted[hidden],
        target[hidden],
        direction=panel.variable == "wind_direction",
    ).numpy()
    baseline_error = physical_error(
        baseline[hidden],
        target[hidden],
        direction=panel.variable == "wind_direction",
    ).numpy()
    timestamps = sample["timestamps"].numpy().astype("datetime64[ns]").astype(str)
    return {
        "schema_version": 1,
        "prediction_postprocessing": prediction_policy(),
        "model": "giano/imputeformer",
        "variable": panel.variable,
        "station": panel.station_ids[global_node],
        "coordinates": {
            "latitude": float(panel.coordinates[global_node, 0]),
            "longitude": float(panel.coordinates[global_node, 1]),
        },
        "split": split,
        "seed": seed,
        "case": asdict(case),
        "sample_index": sample_index,
        "start_index": int(sample["start_index"]),
        "metrics": {
            "hidden_points": int(hidden.sum()),
            "model_mae": float(np.abs(model_error).mean()),
            "model_rmse": float(np.sqrt(np.square(model_error).mean())),
            "interpolation_mae": float(np.abs(baseline_error).mean()),
            "interpolation_rmse": float(np.sqrt(np.square(baseline_error).mean())),
        },
        "series": {
            "timestamps": timestamps.tolist(),
            "ground_truth": _nullable(ground_truth),
            "corrupted": _nullable(corrupted),
            "interpolation": _nullable(interpolation),
            "reconstruction": _nullable(reconstruction),
            "raw_reconstruction": _nullable(
                np.where(
                    visible_values,
                    target_values,
                    np.where(
                        hidden_values, raw_prediction[:, local_node].numpy(), np.nan
                    ),
                )
            ),
            "mask": hidden_values.tolist(),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse reconstruction-example arguments."""
    parser = argparse.ArgumentParser(
        description="Export one masked ground-truth reconstruction example"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/2-processed-v2"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/imputeformer")
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--variable", choices=VARIABLE_TYPE_NAMES, default="temperature"
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mask-type",
        choices=("point", "block", "spatial_block", "terminal", "empirical"),
        default="block",
    )
    parser.add_argument("--mask-parameter", type=float, default=12.0)
    parser.add_argument(
        "--sample-index",
        type=int,
        help="fixed window index; omit to select a random window",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="optional seed for reproducible random window selection",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate the JSON artifact consumed by the visualization notebook."""
    args = parse_args(argv)
    case = BenchmarkCase(cast(SyntheticMaskType, args.mask_type), args.mask_parameter)
    checkpoint = args.checkpoint or imputeformer_checkpoint_path(
        args.checkpoint_dir, args.variable, seed=args.seed
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    device = default_device()
    model, metadata = load_imputeformer_checkpoint(checkpoint, device)
    if metadata.get("variable") != args.variable:
        raise ValueError("checkpoint variable does not match --variable")
    panel = build_spatiotemporal_panel(args.data_dir, args.variable)
    payload = reconstruction_example(
        model,
        metadata,
        panel,
        split=args.split,
        case=case,
        seed=args.seed,
        sample_index=args.sample_index,
        random_seed=args.random_seed,
        device=device,
    )
    payload["checkpoint"] = str(checkpoint)
    payload["training_run"] = metadata.get("training_run")
    payload["code"] = git_provenance(Path.cwd().resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sample_index": payload["sample_index"],
                "station": payload["station"],
                "metrics": payload["metrics"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
