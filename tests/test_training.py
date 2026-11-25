import hashlib
import json

import numpy as np
import pytest
import torch

from giano.train import train, training_device, training_order
from giano.trentino import PROTOCOL
from giano.windows import digest


def test_full_sampling_order_matches_original_experiment():
    hashes = {
        42: "33b2513c76e8fc77eb16a2e096c22b666b1b591437e23810859167ebeb70f180",
        43: "c9b9ecdd2fce116474ad40a31db3c03a9452b880080234915484b1b03e60d057",
    }
    with np.load(PROTOCOL / "train.npz") as data:
        for seed, expected in hashes.items():
            rows = training_order(data["legacy"], seed, 36).ravel()
            assert digest(rows.tolist()) == expected


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_training_preview_exports_new_weights_and_resumes(tmp_path, device):
    from pathlib import Path

    sample = (
        Path(__file__).resolve().parents[1] / "examples/trentino/training-preview.npz"
    )
    cache = tmp_path / "cache"
    for split in ("train", "val"):
        (cache / split).mkdir(parents=True)
    with np.load(sample) as fields:
        for key, array in fields.items():
            split, name = key.split("__")
            np.save(cache / split / f"{name}.npy", array)
    (cache / "recipe.json").write_text(json.dumps({"preview": True, "protocol": 1}))
    run = tmp_path / "trained"
    train(cache, run, device=device)
    report = json.loads((run / "training.json").read_text())
    assert report["device"] == device
    assert report["preview"] is True
    assert report["seeds"]["42"]["history"][0]["loss"] == pytest.approx(
        1.271106, abs=0.005 if report["precision"] == "bf16" else 1e-5
    )
    assert report["seeds"]["43"]["selected_round"] == 1
    ensemble = run / "ensemble.pt"
    assert (
        report["ensemble_sha256"] == hashlib.sha256(ensemble.read_bytes()).hexdigest()
    )
    trained = torch.load(ensemble, weights_only=True)
    initial = torch.load(PROTOCOL / "initial.pt", weights_only=True)
    for member, fresh in zip(trained, initial, strict=True):
        assert all(
            t.device.type == "cpu" and torch.isfinite(t).all() for t in member.values()
        )
        assert any(not torch.equal(member[k], fresh[k]) for k in member)
    if device == "cpu":
        # Checkpoints written by the original CPU trainer lack runtime metadata.
        for checkpoint in run.glob("seed-*.pt"):
            state = torch.load(checkpoint, weights_only=True)
            del state["runtime"]
            torch.save(state, checkpoint)
    train(cache, run, device=device, resume=True)
    for previous, current in zip(
        trained, torch.load(ensemble, weights_only=True), strict=True
    ):
        for name in previous:
            torch.testing.assert_close(previous[name], current[name], rtol=0, atol=0)


def test_automatic_device_and_explicit_unavailable_backend(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert training_device("auto").type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert training_device("auto").type == "mps"
    with pytest.raises(ValueError, match="cuda.*unavailable"):
        training_device("cuda")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert training_device("auto").type == "cpu"
