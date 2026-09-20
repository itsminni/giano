import pytest

from giano.evaluation.campaign import _cached_job, _jobs


def test_downstream_repair_phase_has_separate_outputs_and_all_six_variables(tmp_path):
    jobs = _jobs(tmp_path, phase="downstream")
    assert len(jobs) == 6
    assert all(stage[0].startswith("downstream/") for stage, _ in jobs)
    assert all("corrected_v2_downstream" in str(output) for _, output in jobs)


def test_release_jobs_use_validation_and_all_seeds_without_training(tmp_path):
    jobs = _jobs(tmp_path)
    assert len(jobs) == 24
    for (name, _trainer, arguments), output in jobs:
        assert all(str(seed) in arguments for seed in range(42, 47))
        assert "--data-dir" in arguments
        assert "2-processed-v2" in arguments[arguments.index("--data-dir") + 1]
        assert "corrected_v2_release" in str(output)
        if not name.startswith("downstream"):
            assert arguments[arguments.index("--split") + 1] == "val"


def test_cached_job_reuses_only_matching_completed_outputs(tmp_path):
    output = tmp_path / "result.json"
    calls = []

    def trainer(arguments):
        calls.append(arguments)
        output.write_text("{}")
        return 0

    _, run, arguments = _cached_job(
        ("test", trainer, ["args"]), output, {"data": "hash"}
    )
    assert run(arguments) == 0
    assert run(arguments) == 0
    assert len(calls) == 1
    _, changed, arguments = _cached_job(
        ("test", trainer, ["args"]), output, {"data": "changed"}
    )
    with pytest.raises(ValueError, match="changed"):
        changed(arguments)
    output.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        run(arguments)


def test_cached_job_does_not_overwrite_unverified_output(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("preserve")
    _, run, args = _cached_job(("test", lambda _: 0, []), output, {})
    with pytest.raises(FileExistsError, match="Unverified"):
        run(args)
    assert output.read_text() == "preserve"
