from __future__ import annotations

import torch

from giano.model.export_onnx import _fixed_shapes
from giano.model.imputeformer import ImputeFormerConfig, MeteorologicalImputeFormer


def test_external_gap_gate_matches_full_pytorch_forward() -> None:
    config = ImputeFormerConfig(
        input_dim=12,
        input_embedding_dim=8,
        spatial_embedding_dim=8,
        num_heads=2,
        num_layers=1,
        projection_tokens=4,
        feed_forward_dim=16,
        coordinate_frequencies=2,
        dropout=0.0,
        min_learned_gap=5,
    )
    model = MeteorologicalImputeFormer(config).eval()
    readout = model.readout[-1]
    assert isinstance(readout, torch.nn.Linear)
    assert readout.bias is not None
    with torch.no_grad():
        readout.bias.fill_(0.5)
    features = torch.randn(1, 12, 3, 12)
    node_mask = torch.tensor([[True, True, False]])
    features[..., 3] = 1.0
    features[:, 2:5, 0, 3] = 0.0
    features[:, 4:10, 1, 3] = 0.0
    coordinates = torch.randn(1, 3, 2)
    baseline = torch.randn(1, 12, 3)
    missing = (features[..., 3] < 0.5) & node_mask[:, None, :]
    learned_gap = model._missing_run_lengths(missing) >= config.min_learned_gap

    with torch.no_grad():
        full = model(features, coordinates, node_mask, baseline)
        core = model.forward_core(
            features,
            coordinates,
            node_mask,
            baseline,
            learned_gap,
        )
        blocked = model.forward_core(
            features, coordinates, node_mask, baseline, torch.zeros_like(learned_gap)
        )
        unrestricted = model.forward_core(
            features, coordinates, node_mask, baseline, torch.ones_like(learned_gap)
        )

    assert torch.equal(full[0], core[0])
    assert torch.equal(full[1], core[1])
    assert not core[1][:, 2:5, 0].any()
    assert not core[1][features[..., 3].bool()].any()
    assert not core[1][:, :, 2].any()  # padding never receives a correction
    assert torch.all(core[1][:, 4:10, 1] != 0)
    assert not torch.equal(full[0], baseline)
    assert torch.equal(blocked[0], baseline)
    assert not torch.equal(full[0], blocked[0])
    assert torch.all(unrestricted[1][:, 2:5, 0] != 0)
    assert not torch.equal(full[0], unrestricted[0])


def test_fixed_shapes_record_actual_wind_nodes():
    assert _fixed_shapes(torch.zeros(1, 72, 6, 12)) == {
        "batch": 1,
        "time": 72,
        "nodes": 6,
        "features": 12,
    }
