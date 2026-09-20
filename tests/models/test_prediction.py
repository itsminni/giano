"""The same physical-output contract applies to PyTorch and ONNX clients."""

import numpy as np
import pytest
import torch

from giano.evaluation.gapfill.results import evaluate_model
from giano.meteorology import PHYSICAL_BOUNDS_BY_VARIABLE
from giano.prediction import (
    physical_prediction,
    physical_prediction_numpy,
    prediction_policy,
)
from giano.training_progress import load_training_progress


@pytest.mark.parametrize("variable", PHYSICAL_BOUNDS_BY_VARIABLE)
def test_torch_numpy_decode_parity_and_bounds(variable):
    raw = np.array([[[-1000.0, 1000.0], [0.0, 1.0]]], dtype=np.float32)
    center = np.array([[10.0, 20.0]], dtype=np.float32)
    scale = np.array([[2.0, 3.0]], dtype=np.float32)
    expected = center[:, None, :] + raw * scale[:, None, :]
    if variable == "wind_direction":
        expected %= 360
    else:
        expected = np.clip(expected, *PHYSICAL_BOUNDS_BY_VARIABLE[variable])
    actual = physical_prediction(
        torch.from_numpy(raw),
        torch.from_numpy(center),
        torch.from_numpy(scale),
        variable=variable,
    )
    np.testing.assert_array_equal(actual.numpy(), expected)
    np.testing.assert_array_equal(
        physical_prediction_numpy(raw, center, scale, variable=variable), expected
    )


def test_unconstrained_diagnostic_does_not_hide_raw_humidity_errors():
    raw = torch.tensor([[[-2.0, 2.0]]])
    center = torch.tensor([[50.0, 50.0]])
    scale = torch.tensor([[100.0, 100.0]])
    assert physical_prediction(raw, center, scale, variable="humidity").tolist() == [
        [[0.0, 100.0]]
    ]
    assert physical_prediction(
        raw, center, scale, variable="humidity", constrain=False
    ).tolist() == [[[-150.0, 250.0]]]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_outputs_are_errors_not_silently_clipped(bad):
    raw = np.array([[[bad]]], dtype=np.float32)
    center, scale = np.zeros((1, 1)), np.ones((1, 1))
    with pytest.raises(ValueError, match="Non-finite"):
        physical_prediction_numpy(raw, center, scale, variable="humidity")
    with pytest.raises(ValueError, match="Non-finite"):
        physical_prediction(
            torch.from_numpy(raw),
            torch.from_numpy(center),
            torch.from_numpy(scale),
            variable="humidity",
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_normalization_scale_rejected(bad):
    raw = np.zeros((1, 1, 1))
    center, scale = np.zeros((1, 1)), np.full((1, 1), bad)
    with pytest.raises(ValueError, match="scales"):
        physical_prediction_numpy(raw, center, scale, variable="temperature")
    with pytest.raises(ValueError, match="scales"):
        physical_prediction(
            torch.from_numpy(raw),
            torch.from_numpy(center),
            torch.from_numpy(scale),
            variable="temperature",
        )


def test_resume_rejects_legacy_metrics_without_modifying_checkpoint(tmp_path):
    path = tmp_path / "training.resume.pt"
    payload = {
        "checkpoint_kind": "training_progress",
        "model_class": "Example",
        "variable": "humidity",
        "model_config": {},
        "training_config": {},
    }
    torch.save(payload, path)
    before = path.read_bytes()
    training_dataset = {"sha256": "fixture"}
    with pytest.raises(ValueError, match="validation postprocessing"):
        load_training_progress(
            path,
            torch.device("cpu"),
            model_class="Example",
            variable="humidity",
            model_config={},
            training_config={},
            training_dataset=training_dataset,
        )
    assert path.read_bytes() == before
    payload["validation_postprocessing"] = prediction_policy()
    payload["training_dataset"] = training_dataset
    torch.save(payload, path)
    assert (
        load_training_progress(
            path,
            torch.device("cpu"),
            model_class="Example",
            variable="humidity",
            model_config={},
            training_config={},
            training_dataset=training_dataset,
        )
        == payload
    )


def test_common_evaluator_scores_bounded_physical_predictions():
    class ExtremeModel(torch.nn.Module):
        def forward(self, features, coordinates, node_mask, baseline):
            return torch.full_like(baseline, 1000), torch.zeros_like(baseline)

    sample = {
        "features": torch.zeros(4, 1, 12),
        "coordinates": torch.zeros(1, 2),
        "node_mask": torch.ones(1, dtype=torch.bool),
        "baseline": torch.zeros(4, 1),
        "center": torch.tensor([50.0]),
        "scale": torch.ones(1),
        "baseline_physical": torch.full((4, 1), 50.0),
        "target_physical": torch.full((4, 1), 50.0),
        "evaluation_mask": torch.ones(4, 1, dtype=torch.bool),
    }

    class SingleSampleDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            if index != 0:
                raise IndexError(index)
            return sample

    result = evaluate_model(
        ExtremeModel(),
        torch.utils.data.DataLoader(SingleSampleDataset()),
        torch.device("cpu"),
        "humidity",
        max_batches=1,
    )
    assert result.model_mae == 50
    assert result.model_rmse == 50
    assert result.baseline_mae == 0
