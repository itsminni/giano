"""Package the temperature ONNX model and two demo inputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from giano.evaluation.gapfill.cases import BenchmarkCase, dataset_for_case
from giano.model.export_onnx import _numpy_inputs, _onnx_inputs, _runtime_session
from giano.model.train_imputeformer import TrainingConfig, load_imputeformer_checkpoint
from giano.prediction import prediction_policy
from giano.provenance import file_sha256
from giano.spatiotemporal_dataset import INPUT_FEATURES, build_spatiotemporal_panel


def tensor_payload(value: np.ndarray) -> dict[str, Any]:
    """Serialize a tensor in row-major order."""
    array = np.asarray(value)
    if array.dtype not in (np.dtype("float32"), np.dtype("bool")):
        raise ValueError("Demo tensors must be float32 or bool")
    if not np.isfinite(array).all():
        raise ValueError("Demo tensors must be finite")
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "values": array.ravel().tolist(),
    }


def prepare_bundle(root: Path, output: Path) -> dict[str, Any]:
    """Check ONNX parity and package the demo."""
    if output.exists():
        raise FileExistsError(f"Preserve existing demo assets: {output}")
    checkpoint = (
        root / "checkpoints/corrected_v2_spectral/imputeformer/seed-42/temperature.pt"
    )
    exported = (
        root / "artifacts/models/corrected_v2_release/giano_temperature_seed42.onnx"
    )
    export_metadata = json.loads(exported.with_suffix(".json").read_text())
    if file_sha256(exported) != export_metadata["onnx_sha256"]:
        raise ValueError("ONNX bytes differ from their verified export metadata")
    model, metadata = load_imputeformer_checkpoint(checkpoint, torch.device("cpu"))
    panel = build_spatiotemporal_panel(root / "data/2-processed-v2", "temperature")
    config = TrainingConfig.from_checkpoint(metadata)
    session = _runtime_session(exported)
    examples = []
    for case in (BenchmarkCase("point", 0.3), BenchmarkCase("block", 12)):
        dataset = dataset_for_case(panel, "val", config, case, 42)
        index = min(17, len(dataset) - 1)
        sample = dataset[index]
        batch = {key: value.unsqueeze(0) for key, value in sample.items()}
        inputs = _numpy_inputs(_onnx_inputs(model, batch))
        with torch.no_grad():
            expected = model(
                batch["features"],
                batch["coordinates"],
                batch["node_mask"],
                batch["baseline"],
            )
        actual = session.run(None, inputs)
        error = max(
            float(np.abs(reference.numpy() - value).max())
            for reference, value in zip(expected, actual, strict=True)
        )
        if not np.isfinite(error) or error >= 1e-4:
            raise ValueError(f"Demo input failed ONNX parity: {error}")
        examples.append(
            {
                "name": case.label,
                "split": "val",
                "seed": 42,
                "sample_index": index,
                "station_ids": [
                    panel.station_ids[int(node)] if node >= 0 else None
                    for node in sample["node_indices"]
                ],
                "timestamps": sample["timestamps"]
                .numpy()
                .astype("datetime64[ns]")
                .astype(str)
                .tolist(),
                "inputs": {key: tensor_payload(value) for key, value in inputs.items()},
                "reference_outputs": {
                    key: tensor_payload(value.numpy())
                    for key, value in zip(
                        ("prediction", "residual"), expected, strict=True
                    )
                },
                "center": tensor_payload(sample["center"].numpy()),
                "scale": tensor_payload(sample["scale"].numpy()),
                "ground_truth": tensor_payload(sample["target_physical"].numpy()),
                "ground_truth_available": tensor_payload(
                    (sample["visible_mask"] | sample["evaluation_mask"]).numpy()
                ),
                "evaluation_mask": tensor_payload(sample["evaluation_mask"].numpy()),
                "onnx_max_abs_error": error,
            }
        )

    output.mkdir(parents=True)
    shutil.copy2(exported, output / "giano_temperature_seed42.onnx")
    (output / "examples.json").write_text(
        json.dumps({"schema_version": 1, "examples": examples}, allow_nan=False) + "\n"
    )
    payload = {
        "schema_version": 1,
        "demonstration_only": True,
        "policy_status": "deferred",
        "variable": "temperature",
        "family": "imputeformer",
        "seed": 42,
        "checkpoint_sha256": file_sha256(checkpoint),
        "training_objective": metadata["training_objective"],
        "feature_order": list(INPUT_FEATURES),
        "prediction_postprocessing": prediction_policy(),
        "fixed_shapes": export_metadata["fixed_shapes"],
        "tensor_order": "row-major",
        "normalized_parity_tolerance": 1e-4,
        "browser_tested": False,
        "files": {
            name: {
                "sha256": file_sha256(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in ("giano_temperature_seed42.onnx", "examples.json")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/demo/corrected_v2_release")
    )
    args = parser.parse_args()
    print(json.dumps(prepare_bundle(args.project_root.resolve(), args.output)))
