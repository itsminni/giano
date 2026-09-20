"""Keep model families independent and select the available accelerator."""

import json
import subprocess
import sys

import pytest
import torch

from giano.runtime import default_device


@pytest.mark.parametrize(
    ("module", "forbidden"),
    [
        ("giano.model.train_imputeformer", ["giano.baselines", "giano.downstream"]),
        ("giano.impute", ["giano.baselines", "giano.downstream"]),
        ("giano.baselines.train_bilstm", ["giano.model", "giano.downstream"]),
        (
            "giano.evaluation.gapfill.benchmark_bilstm",
            ["giano.model", "giano.downstream"],
        ),
        ("giano.downstream", ["torch", "giano.model", "giano.baselines"]),
        (
            "giano.evaluation.forecasting.checkpoint_repairer",
            ["giano.model", "giano.baselines"],
        ),
    ],
)
def test_import_does_not_load_unrelated_models(
    module: str, forbidden: list[str]
) -> None:
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import importlib, json, sys; importlib.import_module(sys.argv[1]); "
            "print(json.dumps(sorted(sys.modules)))",
            module,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    loaded = json.loads(result.stdout)
    unexpected = [
        name
        for name in loaded
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
    ]
    assert not unexpected, unexpected


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_device_selection(
    monkeypatch, cuda_available: bool, mps_available: bool, expected: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)
    assert default_device().type == expected
