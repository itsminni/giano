"""Export the true Giano neural core and verify PyTorch/ONNX parity."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from giano.evaluation.gapfill.cases import BenchmarkCase, dataset_for_case
from giano.model.imputeformer import MeteorologicalImputeFormer
from giano.model.train_imputeformer import TrainingConfig, load_imputeformer_checkpoint
from giano.prediction import (
    physical_prediction,
    physical_prediction_numpy,
    prediction_policy,
)
from giano.provenance import file_sha256, git_provenance
from giano.spatiotemporal_dataset import (
    SpatiotemporalPanel,
    build_spatiotemporal_panel,
)

ONNX_INPUTS = ("features", "coordinates", "node_mask", "baseline", "learned_gap")
PARITY_CASES = (
    BenchmarkCase("point", 0.3),
    BenchmarkCase("block", 3),
    BenchmarkCase("block", 12),
    BenchmarkCase("terminal", 24),
)


class _ExportableCore(torch.nn.Module):
    def __init__(self, model: MeteorologicalImputeFormer) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        node_mask: torch.Tensor,
        baseline: torch.Tensor,
        learned_gap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_core(
            features,
            coordinates,
            node_mask,
            baseline,
            learned_gap,
        )


def _batch(
    panel: SpatiotemporalPanel,
    config: TrainingConfig,
    case: BenchmarkCase,
    seed: int,
) -> dict[str, torch.Tensor]:
    dataset = dataset_for_case(panel, "test", config, case, seed)
    return {key: value.unsqueeze(0) for key, value in dataset[0].items()}


def _onnx_inputs(
    model: MeteorologicalImputeFormer,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    missing = (batch["features"][..., 3] < 0.5) & batch["node_mask"][:, None, :]
    learned_gap = model._missing_run_lengths(missing) >= model.config.min_learned_gap
    return (
        batch["features"],
        batch["coordinates"],
        batch["node_mask"],
        batch["baseline"],
        learned_gap,
    )


def _runtime_session(path: Path) -> Any:
    try:
        import onnxruntime as ort  # type: ignore[import-not-found,import-untyped]
    except ImportError as error:
        raise RuntimeError(
            "ONNX parity requires the optional onnx dependency group"
        ) from error
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _numpy_inputs(inputs: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    return {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(ONNX_INPUTS, inputs, strict=True)
    }


def _fixed_shapes(features: torch.Tensor) -> dict[str, int]:
    """Describe the exported tensor, not the configured upper node limit."""
    if features.ndim != 4:
        raise ValueError("ONNX features must have four dimensions")
    return dict(
        zip(("batch", "time", "nodes", "features"), features.shape, strict=True)
    )


def export_and_verify(  # noqa: PLR0914
    checkpoint: Path,
    data_root: Path,
    output: Path,
    *,
    seed: int,
    tolerance: float,
) -> dict[str, Any]:
    """Export fixed-shape core and verify representative and edge cases."""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("ONNX tolerance must be finite and positive")
    model, metadata = load_imputeformer_checkpoint(checkpoint, torch.device("cpu"))
    variable = metadata.get("variable")
    if not isinstance(variable, str):
        raise ValueError("checkpoint variable is invalid")
    config = TrainingConfig.from_checkpoint(metadata)
    panel = build_spatiotemporal_panel(data_root, variable)
    example = _batch(panel, config, PARITY_CASES[0], seed)
    example_inputs = _onnx_inputs(model, example)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _ExportableCore(model).eval(),
        example_inputs,
        output,
        input_names=list(ONNX_INPUTS),
        output_names=["prediction", "residual"],
        opset_version=18,
        dynamo=False,
    )
    session = _runtime_session(output)
    for spec, tensor in zip(session.get_inputs(), example_inputs, strict=True):
        if spec.shape != list(tensor.shape):
            raise RuntimeError(f"ONNX input shape mismatch: {spec.name}")
    parity: list[dict[str, Any]] = []
    worst = 0.0
    for case in PARITY_CASES:
        batch = _batch(panel, config, case, seed)
        inputs = _onnx_inputs(model, batch)
        with torch.no_grad():
            expected = model.forward_core(*inputs)
        actual = session.run(None, _numpy_inputs(inputs))
        prediction_error = float(
            np.max(np.abs(expected[0].numpy() - np.asarray(actual[0])))
        )
        residual_error = float(
            np.max(np.abs(expected[1].numpy() - np.asarray(actual[1])))
        )
        visible = batch["features"][..., 3].bool()
        visible_residual = float(expected[1][visible].abs().max().item())
        physical_expected = physical_prediction(
            expected[0], batch["center"], batch["scale"], variable=variable
        ).numpy()
        physical_actual = physical_prediction_numpy(
            np.asarray(actual[0]),
            batch["center"].numpy(),
            batch["scale"].numpy(),
            variable=variable,
        )
        physical_delta = np.abs(physical_expected - physical_actual)
        if variable == "wind_direction":
            physical_delta = np.abs(
                (physical_expected - physical_actual + 180.0) % 360.0 - 180.0
            )
        worst = max(worst, prediction_error, residual_error)
        parity.append(
            {
                "case": asdict(case),
                "prediction_max_abs_error": prediction_error,
                "residual_max_abs_error": residual_error,
                "visible_residual_max_abs": visible_residual,
                "physical_prediction_max_abs_error": float(physical_delta.max()),
            }
        )
    if worst >= tolerance:
        raise RuntimeError(f"ONNX parity {worst:.8g} exceeds tolerance {tolerance}")
    runtime_inputs = _numpy_inputs(example_inputs)
    for _ in range(3):
        session.run(None, runtime_inputs)
    durations = []
    for _ in range(20):
        started = time.perf_counter()
        session.run(None, runtime_inputs)
        durations.append((time.perf_counter() - started) * 1000.0)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "model": "giano/imputeformer",
        "variable": variable,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "onnx": str(output),
        "opset": 18,
        "fixed_shapes": _fixed_shapes(example_inputs[0]),
        "configured_max_nodes": config.max_nodes,
        "external_gap_gate": True,
        "external_prediction_postprocessing": prediction_policy(),
        "output_units": "normalized; apply external decoding before use",
        "tolerance": tolerance,
        "max_abs_error": worst,
        "parity_cases": parity,
        "onnx_size_bytes": output.stat().st_size,
        "onnx_sha256": file_sha256(output),
        "onnxruntime_cpu_latency_ms": {
            "median": float(np.median(durations)),
            "p95": float(np.quantile(durations, 0.95)),
        },
        "browser_latency_measured": False,
        "code": git_provenance(Path.cwd().resolve()),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse fixed-shape export settings."""
    parser = argparse.ArgumentParser(
        description="Export and verify the true Giano ONNX neural core"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/imputeformer/seed-42/temperature.pt"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/2-processed-v2"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/models/giano_temperature_seed42.onnx"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run export and print its compact verification summary."""
    args = parse_args(argv)
    if args.tolerance <= 0:
        raise ValueError("ONNX tolerance must be positive")
    payload = export_and_verify(
        args.checkpoint,
        args.data_dir,
        args.output,
        seed=args.seed,
        tolerance=args.tolerance,
    )
    print(
        json.dumps(
            {
                "onnx": payload["onnx"],
                "metadata": str(args.output.with_suffix(".json")),
                "max_abs_error": payload["max_abs_error"],
                "onnx_size_bytes": payload["onnx_size_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
