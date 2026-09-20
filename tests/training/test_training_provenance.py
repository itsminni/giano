"""Training must never reuse scores or optimizer state from different data."""

import os
import shutil
from dataclasses import asdict
from functools import partial

import pytest
import torch

from giano.baselines.bilstm import BiLSTMConfig
from giano.baselines.train_bilstm import BiLSTMTrainingConfig, train_bilstm
from giano.evaluation.gapfill.results import EvaluationResult
from giano.model.imputeformer import ImputeFormerConfig
from giano.model.train_imputeformer import (
    TrainingConfig,
    train_variable,
    training_objective,
)
from giano.prediction import prediction_policy
from giano.provenance import training_dataset_identity
from giano.training_progress import load_training_progress


def _source(root):
    directory = root / "train"
    directory.mkdir(parents=True)
    path = directory / "A_temperature_merged.nc"
    path.write_bytes(b"observation=10")
    return path


def test_identity_detects_changes_even_with_same_file_size_and_mtime(tmp_path):
    path = _source(tmp_path)
    original = training_dataset_identity(tmp_path, "temperature")
    stat = path.stat()
    path.write_bytes(b"observation=20")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    assert training_dataset_identity(tmp_path, "temperature") != original


def test_identity_allows_identical_tree_relocation_but_tracks_loading_order(tmp_path):
    root = tmp_path / "original"
    _source(root)
    original = training_dataset_identity(root, "temperature")
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert training_dataset_identity(relocated, "temperature") == original
    (relocated / "train").rename(relocated / "val")
    assert training_dataset_identity(relocated, "temperature") != original


def test_identity_ignores_other_variables_but_rejects_missing_and_duplicate_stations(
    tmp_path,
):
    _source(tmp_path)
    original = training_dataset_identity(tmp_path, "temperature")
    (tmp_path / "train/A_humidity_merged.nc").write_bytes(b"other variable")
    assert training_dataset_identity(tmp_path, "temperature") == original
    with pytest.raises(FileNotFoundError):
        training_dataset_identity(tmp_path / "absent", "temperature")
    (tmp_path / "val").mkdir()
    (tmp_path / "val/A_temperature_merged.nc").write_bytes(b"duplicate")
    with pytest.raises(ValueError, match="Duplicate"):
        training_dataset_identity(tmp_path, "temperature")


@pytest.mark.parametrize("family", ["giano", "bilstm"])
def test_completed_resume_requires_dataset_identity_and_never_overwrites(
    tmp_path, family
):
    source = _source(tmp_path / "data")
    identity = training_dataset_identity(tmp_path / "data", "temperature")
    checkpoint = tmp_path / "completed.pt"
    inputs = (tmp_path / "data", "temperature", checkpoint)
    if family == "giano":
        model_config, config = ImputeFormerConfig(), TrainingConfig()
        model_class = "MeteorologicalImputeFormer"
        trainer = partial(
            train_variable, *inputs, model_config, config, torch.device("cpu")
        )
    else:
        model_config, config = BiLSTMConfig("fair"), BiLSTMTrainingConfig()
        model_class = "MeteorologicalBiLSTM"
        trainer = partial(
            train_bilstm, *inputs, model_config, config, torch.device("cpu")
        )
    score = EvaluationResult(1.0, 2.0, 1.0, 2.0, 1.0, 10)
    payload = {
        "model_class": model_class,
        "variant": "fair",
        "variable": "temperature",
        "model_config": asdict(model_config),
        "training_config": asdict(config),
        "training_objective": training_objective() if family == "giano" else None,
        "validation": asdict(score),
        "validation_postprocessing": prediction_policy(),
        "training_dataset": identity,
    }
    torch.save(payload, checkpoint)
    before = checkpoint.read_bytes()
    assert trainer(resume=True) == score
    with pytest.raises(FileExistsError):
        trainer()
    source.write_bytes(b"observation=20")
    with pytest.raises(ValueError, match="dataset identity"):
        trainer(resume=True)
    assert checkpoint.read_bytes() == before
    source.unlink()
    with pytest.raises(FileNotFoundError):
        trainer(resume=True)


@pytest.mark.parametrize("stored", [None, {"version": "legacy-2d-spectrum"}])
def test_giano_completed_resume_rejects_missing_or_changed_objective(tmp_path, stored):
    _source(tmp_path / "data")
    model_config, config = ImputeFormerConfig(), TrainingConfig()
    checkpoint = tmp_path / "old.pt"
    torch.save(
        {
            "model_class": "MeteorologicalImputeFormer",
            "variable": "temperature",
            "model_config": asdict(model_config),
            "training_config": asdict(config),
            "training_objective": stored,
            "validation_postprocessing": prediction_policy(),
            "training_dataset": training_dataset_identity(
                tmp_path / "data", "temperature"
            ),
        },
        checkpoint,
    )
    before = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="training objective"):
        train_variable(
            tmp_path / "data",
            "temperature",
            checkpoint,
            model_config,
            config,
            torch.device("cpu"),
            resume=True,
        )
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("stored", [None, {"sha256": "other"}])
def test_incomplete_resume_rejects_absent_or_changed_identity(tmp_path, stored):
    path = tmp_path / "run.resume.pt"
    torch.save(
        {
            "checkpoint_kind": "training_progress",
            "model_class": "Example",
            "variable": "humidity",
            "model_config": {},
            "training_config": {},
            "validation_postprocessing": prediction_policy(),
            "training_dataset": stored,
        },
        path,
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="dataset identity"):
        load_training_progress(
            path,
            torch.device("cpu"),
            model_class="Example",
            variable="humidity",
            model_config={},
            training_config={},
            training_dataset={"sha256": "actual"},
        )
    assert path.read_bytes() == before
