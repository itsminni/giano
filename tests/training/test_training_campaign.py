"""The corrected-data queue must resume safely and stop after any failed stage."""

import json
import subprocess
import sys

import pytest

from giano.training_campaign import (
    SEEDS,
    VARIABLES,
    _campaign_lock,
    _freeze_manifest,
    _run_stages,
    _stages,
)
from giano.variables import VARIABLE_TYPE_NAMES


def test_campaign_covers_all_variables_and_five_seeds_with_resume(tmp_path):
    assert set(VARIABLES) == set(VARIABLE_TYPE_NAMES)
    assert SEEDS == (42, 43, 44, 45, 46)
    stages = _stages(tmp_path, tmp_path / "frozen.yaml")
    assert [stage[0] for stage in stages] == ["giano", "fair", "legacy_retrained"]
    for _, _, arguments in stages:
        assert "--resume" in arguments
        assert "--quick" not in arguments
        assert arguments[arguments.index("--data-dir") + 1] == str(
            tmp_path / "data/2-processed-v2"
        )
        assert all(variable in arguments for variable in VARIABLES)
        assert all(str(seed) in arguments for seed in SEEDS)


def test_campaign_lock_rejects_duplicate_writer_and_releases_after_interrupt(tmp_path):
    path = tmp_path / "campaign.lock"
    with pytest.raises(KeyboardInterrupt), _campaign_lock(path):
        with pytest.raises(RuntimeError, match="already running"), _campaign_lock(path):
            pytest.fail("second writer acquired the lock")
        raise KeyboardInterrupt
    with _campaign_lock(path):
        assert path.is_file()


def test_campaign_lock_excludes_another_process(tmp_path):
    path = tmp_path / "campaign.lock"
    script = """
import sys
from pathlib import Path
from giano.training_campaign import _campaign_lock
try:
    with _campaign_lock(Path(sys.argv[1])):
        pass
except RuntimeError as error:
    print(error)
    sys.exit(23)
"""
    with _campaign_lock(path):
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script, str(path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    assert result.returncode == 23, result.stderr
    assert "already running" in result.stdout


def test_campaign_lock_is_released_after_process_crash(tmp_path):
    path = tmp_path / "campaign.lock"
    script = """
import os
import sys
from pathlib import Path
from giano.training_campaign import _campaign_lock
with _campaign_lock(Path(sys.argv[1])):
    os._exit(23)
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 23, result.stderr
    with _campaign_lock(path):
        assert path.is_file()


def test_campaign_manifest_refuses_changed_recipe_without_overwriting(tmp_path):
    path = tmp_path / "manifest.json"
    inputs = {"sources": {"model.py": "sha-a"}}
    _freeze_manifest(path, inputs, tmp_path)
    before = path.read_bytes()
    _freeze_manifest(path, inputs, tmp_path)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="inputs changed"):
        _freeze_manifest(path, {"sources": {"model.py": "sha-b"}}, tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("code", [1, 130])
def test_campaign_stops_queue_when_trainer_fails_or_is_interrupted(tmp_path, code):
    calls = []

    def trainer(arguments):
        calls.append(arguments)
        return code

    status_path = tmp_path / "status.json"
    assert (
        _run_stages(
            [("first", trainer, ["first"]), ("second", trainer, ["second"])],
            status_path,
            lambda: None,
        )
        == code
    )
    assert calls == [["first"]]
    status = json.loads(status_path.read_text())
    assert status["state"] == ("interrupted" if code == 130 else "failed")
    assert status["stages"] == {"first": code}


def test_campaign_does_not_train_after_source_change(tmp_path):
    def verify():
        raise ValueError("changed sources")

    def trainer(_arguments):
        pytest.fail("must not start training after source change")

    status_path = tmp_path / "status.json"
    assert _run_stages([("giano", trainer, [])], status_path, verify) == 1
    assert json.loads(status_path.read_text())["state"] == "failed"


def test_campaign_records_success_only_after_every_stage(tmp_path):
    calls = []

    def trainer(arguments):
        calls.append(arguments)
        return 0

    status_path = tmp_path / "status.json"
    assert (
        _run_stages(
            [("giano", trainer, ["giano"]), ("fair", trainer, ["fair"])],
            status_path,
            lambda: None,
        )
        == 0
    )
    assert calls == [["giano"], ["fair"]]
    status = json.loads(status_path.read_text())
    assert status["state"] == "complete"
    assert status["stages"] == {"giano": 0, "fair": 0}
