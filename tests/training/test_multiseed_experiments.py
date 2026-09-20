from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from giano.baselines import train_bilstm as bilstm_training
from giano.baselines.bilstm import BiLSTMConfig
from giano.baselines.train_bilstm import BiLSTMTrainingConfig
from giano.evaluation.gapfill import benchmark_bilstm, benchmark_imputeformer
from giano.evaluation.gapfill.cases import BenchmarkCase, BenchmarkProtocol
from giano.evaluation.gapfill.results import EvaluationResult
from giano.experiments import (
    bilstm_checkpoint_path,
    experiment_seeds,
    imputeformer_checkpoint_path,
)
from giano.model import train_imputeformer as imputeformer_training
from giano.model.imputeformer import ImputeFormerConfig
from giano.model.train_imputeformer import TrainingConfig


def _result() -> EvaluationResult:
    return EvaluationResult(1.0, 2.0, 1.5, 2.5, 1.0, 10)


def _protocol() -> BenchmarkProtocol:
    return BenchmarkProtocol(
        split="test",
        seeds=(42,),
        max_batches=1,
        cases=(BenchmarkCase("block", 6),),
    )


def test_experiment_seeds_and_checkpoint_paths_are_collision_free(
    tmp_path: Path,
) -> None:
    assert experiment_seeds(42, None) == (42,)
    assert experiment_seeds(42, (43, 44)) == (43, 44)
    with pytest.raises(ValueError, match="unique"):
        experiment_seeds(42, (43, 43))
    with pytest.raises(ValueError, match="between"):
        experiment_seeds(42, (-1,))

    assert (
        imputeformer_checkpoint_path(tmp_path, "temperature", seed=43)
        == tmp_path / "seed-43/temperature.pt"
    )
    assert (
        bilstm_checkpoint_path(tmp_path, "fair", "temperature", seed=43)
        == tmp_path / "fair/seed-43/temperature.pt"
    )


def test_imputeformer_training_propagates_each_seed_to_a_distinct_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, int]] = []

    monkeypatch.setattr(
        imputeformer_training,
        "_read_config",
        lambda _path: (ImputeFormerConfig(), TrainingConfig()),
    )
    monkeypatch.setattr(
        imputeformer_training, "default_device", lambda: torch.device("cpu")
    )

    def fake_train_variable(
        _data_root,
        _variable,
        output_path,
        _model_config,
        config,
        default_device,
        *,
        resume=False,
        progress_every=100,
    ):
        assert resume is False
        assert progress_every == 100
        calls.append((output_path, config.seed))
        return _result()

    monkeypatch.setattr(imputeformer_training, "train_variable", fake_train_variable)

    assert (
        imputeformer_training.main(
            [
                "--variables",
                "temperature",
                "--checkpoint-dir",
                str(tmp_path),
                "--seeds",
                "43",
                "44",
            ]
        )
        == 0
    )
    assert calls == [
        (tmp_path / "seed-43/temperature.pt", 43),
        (tmp_path / "seed-44/temperature.pt", 44),
    ]


def test_bilstm_training_propagates_each_seed_to_a_distinct_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, int]] = []

    monkeypatch.setattr(
        bilstm_training,
        "read_bilstm_config",
        lambda _path, variant: (
            BiLSTMConfig(variant),
            BiLSTMTrainingConfig(),
        ),
    )
    monkeypatch.setattr(bilstm_training, "default_device", lambda: torch.device("cpu"))

    def fake_train_bilstm(
        _data_root,
        _variable,
        output_path,
        _model_config,
        config,
        default_device,
        *,
        resume=False,
        progress_every=100,
    ):
        assert resume is False
        assert progress_every == 100
        calls.append((output_path, config.seed))
        return _result()

    monkeypatch.setattr(bilstm_training, "train_bilstm", fake_train_bilstm)

    assert (
        bilstm_training.main(
            [
                "--variables",
                "temperature",
                "--variants",
                "fair",
                "--checkpoint-dir",
                str(tmp_path),
                "--seeds",
                "43",
                "44",
            ]
        )
        == 0
    )
    assert calls == [
        (tmp_path / "fair/seed-43/temperature.pt", 43),
        (tmp_path / "fair/seed-44/temperature.pt", 44),
    ]


@pytest.mark.parametrize(
    ("parse_args", "protocol_with_overrides"),
    [
        (
            benchmark_imputeformer.parse_args,
            benchmark_imputeformer.protocol_with_overrides,
        ),
        (benchmark_bilstm.parse_args, benchmark_bilstm.protocol_with_overrides),
    ],
)
def test_paired_benchmark_seeds_are_explicit_and_cannot_mix_with_mask_only_seeds(
    parse_args,
    protocol_with_overrides,
) -> None:
    args = parse_args(["--training-seeds", "43", "44"])
    assert protocol_with_overrides(_protocol(), args).seeds == (43, 44)

    conflicting = parse_args(["--seeds", "7", "8", "--training-seeds", "43", "44"])
    with pytest.raises(ValueError, match="cannot be combined"):
        protocol_with_overrides(_protocol(), conflicting)


def test_paired_benchmark_rejects_checkpoint_with_wrong_training_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "seed-43/temperature.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    metadata = {
        "variable": "temperature",
        "training_config": asdict(TrainingConfig(seed=42)),
    }

    monkeypatch.setattr(
        benchmark_imputeformer, "default_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(
        benchmark_imputeformer,
        "load_benchmark_protocol",
        lambda _path: _protocol(),
    )
    monkeypatch.setattr(
        benchmark_imputeformer,
        "build_spatiotemporal_panel",
        lambda _root, _variable: object(),
    )
    monkeypatch.setattr(
        benchmark_imputeformer,
        "load_imputeformer_checkpoint",
        lambda _path, default_device: (object(), metadata),
    )

    with pytest.raises(ValueError, match="declares training seed 42, expected 43"):
        benchmark_imputeformer.main(
            [
                "--variables",
                "temperature",
                "--checkpoint-dir",
                str(tmp_path),
                "--training-seeds",
                "43",
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
