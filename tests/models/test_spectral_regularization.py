"""Mask-aware temporal prior: gradients, station invariance and circular data."""

from dataclasses import replace

import pytest
import torch

from giano.model import train_imputeformer as training
from giano.model.imputeformer import temporal_spectral_regularization


def _inputs():
    generator = torch.Generator().manual_seed(42)
    prediction = torch.randn(2, 9, 4, generator=generator, dtype=torch.float64)
    target = torch.randn(2, 9, 4, generator=generator, dtype=torch.float64)
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    visible = (torch.rand(2, 9, 4, generator=generator) > 0.4) & valid[:, None, :]
    visible[0, 1:3, 0] = False
    target[~visible] = float("nan")
    return prediction, target, visible, valid


@pytest.mark.parametrize("direction", [False, True])
def test_prior_uses_visible_values_and_predictions_at_every_missing_entry(direction):
    prediction, target, visible, valid = _inputs()
    actual = temporal_spectral_regularization(
        prediction, target, visible_mask=visible, node_mask=valid, direction=direction
    )
    per_station = []
    for batch, node in valid.nonzero().tolist():
        series = torch.where(
            visible[batch, :, node], target[batch, :, node], prediction[batch, :, node]
        )
        if direction:
            series = torch.stack(
                ((torch.pi * series).sin(), (torch.pi * series).cos()), -1
            )
        spectrum = torch.fft.rfft(series, dim=0, norm="ortho")
        magnitude = (
            torch.linalg.vector_norm(spectrum, dim=-1) if direction else spectrum.abs()
        )
        per_station.append(magnitude.mean())
    torch.testing.assert_close(actual, torch.stack(per_station).mean())

    changed_target = target.masked_fill(~visible, 10000.0)
    changed = temporal_spectral_regularization(
        prediction,
        changed_target,
        visible_mask=visible,
        node_mask=valid,
        direction=direction,
    )
    torch.testing.assert_close(changed, actual, rtol=0, atol=0)


@pytest.mark.parametrize("direction", [False, True])
def test_prior_is_invariant_to_station_order_and_nan_padding(direction):
    prediction, target, visible, valid = _inputs()
    prediction.requires_grad_()
    original = temporal_spectral_regularization(
        prediction, target, visible_mask=visible, node_mask=valid, direction=direction
    )
    (gradient,) = torch.autograd.grad(original, prediction)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = prediction.detach()[:, :, permutation].requires_grad_()
    permuted_loss = temporal_spectral_regularization(
        permuted,
        target[:, :, permutation],
        visible_mask=visible[:, :, permutation],
        node_mask=valid[:, permutation],
        direction=direction,
    )
    (permuted_gradient,) = torch.autograd.grad(permuted_loss, permuted)
    torch.testing.assert_close(permuted_loss, original)
    torch.testing.assert_close(permuted_gradient, gradient[:, :, permutation])

    padded = torch.nn.functional.pad(prediction.detach(), (0, 3), value=float("nan"))
    padded.requires_grad_()
    padded_loss = temporal_spectral_regularization(
        padded,
        torch.nn.functional.pad(target, (0, 3), value=float("nan")),
        visible_mask=torch.nn.functional.pad(visible, (0, 3), value=True),
        node_mask=torch.nn.functional.pad(valid, (0, 3)),
        direction=direction,
    )
    (padded_gradient,) = torch.autograd.grad(padded_loss, padded)
    torch.testing.assert_close(padded_loss, original)
    torch.testing.assert_close(padded_gradient[:, :, :4], gradient)
    assert torch.count_nonzero(padded_gradient[:, :, 4:]) == 0
    assert torch.isfinite(padded_gradient).all()


@pytest.mark.parametrize("direction", [False, True])
def test_prior_backpropagates_through_natural_and_synthetic_gaps_only(direction):
    prediction, target, visible, valid = _inputs()
    prediction.requires_grad_()
    loss = temporal_spectral_regularization(
        prediction, target, visible_mask=visible, node_mask=valid, direction=direction
    )
    (gradient,) = torch.autograd.grad(loss, prediction)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[visible]) == 0
    assert torch.count_nonzero(gradient.masked_select(~valid[:, None, :])) == 0
    # Adjacent hidden slots stand for synthetic and natural gaps respectively.
    assert gradient[0, 1, 0].abs() > 1e-8
    assert gradient[0, 2, 0].abs() > 1e-8
    assert torch.autograd.gradcheck(
        lambda values: temporal_spectral_regularization(
            values, target, visible_mask=visible, node_mask=valid, direction=direction
        ),
        (prediction,),
    )


@pytest.mark.parametrize("direction", [False, True])
def test_empty_valid_station_set_has_zero_finite_loss_and_gradients(direction):
    prediction = torch.full((1, 3, 2), float("nan"), requires_grad=True)
    loss = temporal_spectral_regularization(
        prediction,
        torch.full_like(prediction, float("nan")),
        visible_mask=torch.ones_like(prediction, dtype=torch.bool),
        node_mask=torch.zeros(1, 2, dtype=torch.bool),
        direction=direction,
    )
    loss.backward()
    assert loss.item() == 0.0
    assert prediction.grad is not None
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_direction_prior_is_wrap_and_origin_invariant():
    prediction, target, visible, valid = _inputs()
    original = temporal_spectral_regularization(
        prediction, target, visible_mask=visible, node_mask=valid, direction=True
    )
    for shift in (2.0, -4.0, 0.37):
        changed = temporal_spectral_regularization(
            prediction + shift,
            target + shift,
            visible_mask=visible,
            node_mask=valid,
            direction=True,
        )
        torch.testing.assert_close(changed, original)
    wraps = torch.arange(prediction.numel()).reshape_as(prediction) % 3 * 2
    locally_wrapped = temporal_spectral_regularization(
        prediction + wraps,
        target - wraps,
        visible_mask=visible,
        node_mask=valid,
        direction=True,
    )
    torch.testing.assert_close(locally_wrapped, original)


@pytest.mark.parametrize("length", [1, 8, 9])
def test_prior_supports_single_station_and_short_or_odd_sequences(length):
    prediction = torch.ones(1, length, 1, requires_grad=True)
    loss = temporal_spectral_regularization(
        prediction,
        torch.zeros_like(prediction),
        visible_mask=torch.zeros_like(prediction, dtype=torch.bool),
        node_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize("bad_shape", ["target", "visible", "nodes"])
def test_prior_rejects_misaligned_shapes(bad_shape):
    prediction, target, visible, valid = _inputs()
    if bad_shape == "target":
        target = target[:, :, :1]
    elif bad_shape == "visible":
        visible = visible[:, :, :1]
    else:
        valid = valid[:, :1]
    with pytest.raises(ValueError, match="shape"):
        temporal_spectral_regularization(
            prediction, target, visible_mask=visible, node_mask=valid
        )


def test_training_loss_skips_disabled_spectrum_and_ignores_legacy_context(monkeypatch):
    prediction = torch.tensor([[[1.0], [2.0], [3.0]]], requires_grad=True)
    residual = prediction * 0.1
    batch = {
        "target": torch.zeros_like(prediction),
        "visible_mask": torch.tensor([[[True], [False], [False]]]),
        "evaluation_mask": torch.tensor([[[False], [True], [False]]]),
        "node_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    monkeypatch.setattr(
        training,
        "temporal_spectral_regularization",
        lambda *args, **kwargs: pytest.fail("disabled prior must not execute FFT"),
    )
    config = training.TrainingConfig(low_rank_weight=0.0, residual_weight=0.5)
    expected = prediction[0, 1, 0] + 0.5 * residual[0, 1, 0]
    for context in (0.0, 100.0):
        actual = training._loss(
            batch,
            prediction,
            residual,
            replace(config, context_weight=context),
            direction=False,
        )
        torch.testing.assert_close(actual, expected)


def test_training_loss_passes_visible_mask_and_direction_to_prior(monkeypatch):
    prediction = torch.ones(1, 3, 1)
    residual = torch.zeros_like(prediction)
    batch = {
        "target": torch.zeros_like(prediction),
        "visible_mask": torch.tensor([[[True], [False], [False]]]),
        "evaluation_mask": torch.tensor([[[False], [True], [False]]]),
        "node_mask": torch.ones(1, 1, dtype=torch.bool),
    }

    def checked_prior(prediction, target, *, visible_mask, node_mask, direction):
        assert visible_mask is batch["visible_mask"]
        assert node_mask is batch["node_mask"]
        assert direction is True
        return prediction.new_tensor(2.0)

    monkeypatch.setattr(training, "temporal_spectral_regularization", checked_prior)
    loss = training._loss(
        batch,
        prediction,
        residual,
        training.TrainingConfig(low_rank_weight=0.1),
        direction=True,
    )
    torch.testing.assert_close(loss, torch.tensor(1.2))


@pytest.mark.parametrize(
    "weight", ["low_rank_weight", "context_weight", "residual_weight"]
)
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_training_rejects_invalid_loss_weights(weight, value):
    with pytest.raises(ValueError, match="Loss weights"):
        replace(training.TrainingConfig(), **{weight: value}).validate()
