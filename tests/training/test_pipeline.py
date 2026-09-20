from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from giano import pipeline


def test_pipeline_uses_and_removes_transient_merged_directory(
    monkeypatch,
) -> None:
    commands: list[tuple[str, list[str]]] = []

    def record_step(step_name: str, command: list[str]) -> None:
        commands.append((step_name, command))

    monkeypatch.setattr(pipeline, "_run_step", record_step)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "giano",
            "--config",
            str(Path(__file__).resolve().parents[2] / "config.yaml"),
            "--skip-train",
            "--skip-benchmark",
            "--skip-imputation",
        ],
    )

    assert pipeline.main() == 0
    assert [name for name, _command in commands] == [
        "dataset unification",
        "dataset publication",
    ]

    unify_command = commands[0][1]
    split_command = commands[1][1]
    merged_dir = Path(unify_command[unify_command.index("--out-dir") + 1])
    source_dir = Path(split_command[split_command.index("--source-folder") + 1])
    assert merged_dir == source_dir
    assert not merged_dir.exists()


def test_pipeline_uses_the_same_explicit_weights_for_train_benchmark_and_impute(
    monkeypatch,
) -> None:
    commands: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        pipeline, "_run_step", lambda name, command: commands.append((name, command))
    )
    assert (
        pipeline.main(
            [
                "--config",
                str(Path(__file__).resolve().parents[2] / "config.yaml"),
                "--skip-unify",
                "--skip-split",
            ]
        )
        == 0
    )
    assert len(commands) == 3
    for _, command in commands:
        checkpoint_dir = command[command.index("--checkpoint-dir") + 1]
        assert (
            Path(checkpoint_dir) == pipeline.PROJECT_ROOT / "checkpoints/imputeformer"
        )
        assert "--data-dir" in command
        assert Path(command[command.index("--data-dir") + 1]) == (
            pipeline.PROJECT_ROOT / "data/2-processed-v2"
        )


@pytest.mark.parametrize(
    "module",
    [
        "giano.model.train_imputeformer",
        "giano.baselines.train_bilstm",
        "giano.model.export_onnx",
        "giano.evaluation.gapfill.benchmark_imputeformer",
        "giano.evaluation.gapfill.benchmark_bilstm",
        "giano.evaluation.gapfill.analyze",
        "giano.evaluation.gapfill.detailed",
        "giano.evaluation.gapfill.reconstruction_example",
        "giano.evaluation.forecasting.run",
    ],
)
def test_cli_defaults_use_release_data(module: str) -> None:
    parse_args = importlib.import_module(module).parse_args
    assert parse_args([]).data_dir == Path("data/2-processed-v2")
    assert parse_args(["--data-dir", "custom/data"]).data_dir == Path("custom/data")


def test_pipeline_data_fallback_and_override(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    assert (
        pipeline._processed_path(config)
        == pipeline.PROJECT_ROOT / "data/2-processed-v2"
    )
    config.write_text("paths:\n  processed: custom/data\n", encoding="utf-8")
    assert pipeline._processed_path(config) == pipeline.PROJECT_ROOT / "custom/data"


def test_pipeline_rejects_only_one_skipped_preprocessing_step(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "giano",
            "--config",
            str(Path(__file__).resolve().parents[2] / "config.yaml"),
            "--skip-unify",
            "--skip-train",
            "--skip-benchmark",
            "--skip-imputation",
        ],
    )

    assert pipeline.main() == 2
