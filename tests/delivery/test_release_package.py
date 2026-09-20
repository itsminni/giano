"""Local release integrity and safety checks use tiny, artifact-free fixtures."""

import hashlib
import importlib.util
import json
import stat
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_release", Path(__file__).resolve().parents[2] / "tools/prepare_release.py"
)
assert spec is not None and spec.loader is not None
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.mark.parametrize(
    "name", ["../x", "/x", "a/../b", "a//b", "./a", "C:/x", "a\\b", "a\nx", "", "."]
)
def test_release_rejects_unsafe_paths(name):
    with pytest.raises(ValueError, match="Unsafe"):
        release.safe_name(name)


def test_release_rejects_symlinks(tmp_path):
    (tmp_path / "original").write_text("data")
    (tmp_path / "link").symlink_to(tmp_path / "original")
    with pytest.raises(ValueError, match="Symlink"):
        release.local_path(tmp_path, "link")


@pytest.mark.parametrize(
    "name", [".env", "conf/.env.local", "id_ed25519", "credentials.json", "key.pem"]
)
def test_release_rejects_private_source_names(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not a real secret")
    with pytest.raises(ValueError, match="private"):
        release.source_guard(name, path)


@pytest.mark.parametrize("problem", [None, "changed", "missing", "extra"])
def test_release_inventory_is_exact(tmp_path, problem):
    payload = b"evidence"
    path = tmp_path / "evidence.json"
    path.write_bytes(payload)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": {"evidence.json": hashlib.sha256(payload).hexdigest()}})
    )
    if problem == "changed":
        path.write_bytes(b"modified")
    elif problem == "missing":
        path.unlink()
    elif problem == "extra":
        (tmp_path / "extra.txt").write_text("unexpected")
    if problem:
        with pytest.raises(ValueError):
            release.verify_inventory(tmp_path)
    else:
        assert release.verify_inventory(tmp_path) == 1


@pytest.mark.parametrize(
    "problem", [None, "changed", "size", "extra", "unsafe", "symlink"]
)
def test_release_zip_verification(tmp_path, problem):
    path = tmp_path / "candidate.zip"
    payload = b"source"
    name = "../outside" if problem == "unsafe" else "project/code.py"
    manifest = {
        "schema_version": 1,
        "status": "release",
        "files": {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": 99 if problem == "size" else len(payload),
            },
        },
    }
    with ZipFile(path, "w") as archive:
        info = ZipInfo(name)
        if problem == "symlink":
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, b"change" if problem == "changed" else payload)
        archive.writestr("manifest.json", json.dumps(manifest))
        if problem == "extra":
            archive.writestr("unexpected.txt", "extra")
    if problem:
        with pytest.raises(ValueError):
            release.verify_archive(path)
    else:
        result = release.verify_archive(path)
        assert result["files"] == 2
        assert result["status"] == "verified"


@pytest.mark.parametrize("problem", [None, "dtype", "shape", "nonfinite"])
def test_release_tensor_contract(problem):
    record = {"dtype": "float32", "shape": [1, 2], "values": [1.0, 2.0]}
    if problem == "dtype":
        record["dtype"] = "int64"
    elif problem == "shape":
        record["shape"] = [1, 3]
    elif problem == "nonfinite":
        record["values"] = [1.0, float("nan")]
    if problem:
        with pytest.raises(ValueError):
            release.tensor(record)
    else:
        assert np.array_equal(release.tensor(record), [[1.0, 2.0]])


def test_release_profile_never_selects_or_approves_policy():
    root = Path(__file__).resolve().parents[2]
    cfg = release.profile(root)
    assert cfg["data"] == "data/2-processed-v2"
    assert cfg["presentation"]["operational_policy"] == "deferred"
    assert cfg["name"] == "giano-1.0.0"


def test_release_bundle_does_not_overwrite(tmp_path):
    output = tmp_path / "existing.zip"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        release.bundle(tmp_path, output)
    assert output.read_bytes() == b"preserve"


@pytest.mark.parametrize("mutate_during_check", [False, True])
def test_release_bundle_preserves_verified_source_bytes(
    tmp_path, monkeypatch, mutate_during_check
):
    source = tmp_path / "code.py"
    source.write_text("original")
    report = tmp_path / "registry.json"
    report.write_text('{"checkpoints": []}')
    cfg = {key: key for key in release.PATH_KEYS}
    cfg["name"] = "giano-1.0.0"
    cfg["checkpoint_report"] = report.name
    monkeypatch.setattr(release, "profile", lambda _: cfg)
    monkeypatch.setattr(release, "source_paths", lambda _: ["code.py"])
    monkeypatch.setattr(release, "demo", lambda _: {"status": "passed"})
    monkeypatch.setattr(
        release, "git_provenance", lambda _: {"commit": None, "dirty": True}
    )

    def check(*args, **kwargs):
        if mutate_during_check:
            source.write_text("modified")
        return {"status": "verified"}

    monkeypatch.setattr(release, "check", check)
    output = tmp_path / "candidate.zip"
    if mutate_during_check:
        with pytest.raises(ValueError, match="mismatch"):
            release.bundle(tmp_path, output)
        assert not output.exists()
    else:
        result = release.bundle(tmp_path, output)
        assert result["status"] == "verified"
        with ZipFile(output) as archive:
            assert archive.read("project/code.py") == b"original"


def test_release_zip_rejects_duplicate_members(tmp_path):
    path = tmp_path / "duplicate.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("same.txt", "first")
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("same.txt", "second")
    with pytest.raises(ValueError, match="Duplicate"):
        release.verify_archive(path)


def test_artifacts_bundle_uses_checkout_paths_without_source(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoints/model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    history = tmp_path / "history/index.json"
    history.parent.mkdir()
    history.write_text("{}")
    pilot = tmp_path / "external_evaluation/benchmark.json"
    pilot.parent.mkdir()
    pilot.write_text('{"results": []}')
    report = tmp_path / "registry.json"
    report.write_text(json.dumps({"checkpoints": [{"path": "checkpoints/model.pt"}]}))
    cfg = {key: key for key in release.PATH_KEYS}
    cfg.update(name="giano-1.0.0", checkpoint_report=report.name)
    monkeypatch.setattr(release, "profile", lambda _: cfg)
    monkeypatch.setattr(release, "check", lambda *a, **kw: {"status": "verified"})
    monkeypatch.setattr(release, "demo", lambda _: {"status": "passed"})
    monkeypatch.setattr(release, "git_provenance", lambda _: {})

    def no_source_scan(_):
        pytest.fail("Artifact archives must not scan the source tree")

    monkeypatch.setattr(release, "source_paths", no_source_scan)
    output = tmp_path / "artifacts.zip"
    release.bundle(tmp_path, output, artifacts_only=True, include_history=True)
    with ZipFile(output) as archive:
        assert archive.read("checkpoints/model.pt") == b"weights"
        assert archive.read("history/index.json") == b"{}"
        assert archive.read("external_evaluation/benchmark.json") == pilot.read_bytes()
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["includes"]["source"] is False
        assert manifest["includes"]["full_history"] is True
        assert not any(name.startswith("project/") for name in archive.namelist())
    with pytest.raises(ValueError, match="cannot include the wiki"):
        release.bundle(
            tmp_path, tmp_path / "invalid.zip", artifacts_only=True, wiki_root=tmp_path
        )
