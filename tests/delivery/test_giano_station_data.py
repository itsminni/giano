"""The station presentation is a Giano-only projection."""

import importlib.util
import json
from pathlib import Path

import pytest

from giano.provenance import file_sha256

spec = importlib.util.spec_from_file_location(
    "giano_station_data",
    Path(__file__).resolve().parents[2] / "tools/prepare_giano_station_data.py",
)
assert spec is not None and spec.loader is not None
presentation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(presentation)


def _inputs():
    metrics = {"mae": 1.0, "mean_case_seed_rmse": 1.0, "case_seed_groups": 65}
    results = {
        "split": "val",
        "seeds": [42, 43, 44, 45, 46],
        "variables": {
            "temperature": {
                "unit": "degC",
                "scores": {
                    "giano/imputeformer": metrics,
                    "bilstm/fair": metrics,
                    "interpolation": metrics,
                },
                "original_2025_supported": True,
                "giano_mae_reduction_percent": 0,
            }
        },
        "conclusions": "comparison text that must not be copied",
    }
    station_metrics = {
        "split": "val",
        "aggregation": "equal weight to available case/seed groups",
        "available_case_seed_groups": 1,
        "expected_case_seed_groups": 65,
        "complete_case_seed_coverage": False,
        "hidden_evaluations_including_repeats": 12,
        "scores": {"giano": metrics, "bilstm_fair": metrics, "interpolation": metrics},
        "case_seed_scores": [
            {
                "seed": 42,
                "mask_type": "block",
                "mask_parameter": 12,
                "n_hidden": 12,
                "giano_mae": 1.0,
                "giano_rmse": 1.0,
                "bilstm_fair_mae": 1.0,
                "interpolation_rmse": 1.0,
            }
        ],
    }
    hidden = [20 <= i < 32 for i in range(72)]
    unknown = [i == 5 for i in range(72)]
    target = [None if unknown[i] else float(i) for i in range(72)]
    example = {
        "station": "T0001",
        "variable": "temperature",
        "unit": "degC",
        "split": "val",
        "seed": 42,
        "sample_index": 1,
        "checkpoint_sha256": "test",
        "case": {"mask_type": "block", "requested_length_hours": 12},
        "metrics": {
            "hidden_points": 12,
            "giano_mae": 1.0,
            "giano_rmse": 1.0,
            "interpolation_mae": 1.0,
        },
        "series": {
            "timestamps": [
                f"2020-01-{1 + i // 24:02d}T{i % 24:02d}:00:00" for i in range(72)
            ],
            "timestamp_timezone": "not established",
            "ground_truth": target,
            "observed_with_mask": [
                None if hidden[i] else v for i, v in enumerate(target)
            ],
            "reconstruction": [
                v + 1 if hidden[i] and v is not None else v
                for i, v in enumerate(target)
            ],
            "interpolation": target,
            "synthetic_mask": hidden,
            "ground_truth_unavailable": unknown,
        },
        "extra": "legacy baseline",
    }
    links = {
        "json": "examples/T0001/temperature.json",
        "image": "images/T0001/temperature.png",
    }
    card = {
        "id": "T0001",
        "name": "Station",
        "geometry": {"type": "Point", "coordinates": [11.0, 46.0]},
        "has_local_data": True,
        "has_example": True,
        "has_benchmark_results": True,
        "variables": {
            "temperature": {
                "unit": "degC",
                "data_coverage": {"observed_hours": 100},
                "benchmark": station_metrics,
                "example": links,
            }
        },
        "conclusions": "comparison text",
    }
    return results, card, example


def test_projection_keeps_giano_values_without_nested_research_fields():
    results, card, example = _inputs()
    before = json.dumps((results, card, example))
    projected = (
        presentation.project_results(results),
        presentation.project_card(card),
        presentation.project_example(example),
    )
    assert not presentation.FORBIDDEN.search(json.dumps(projected))
    assert projected[0]["variables"]["temperature"]["metrics"]["mae"] == 1.0
    assert (
        projected[1]["variables"]["temperature"]["giano_results"][
            "complete_case_seed_coverage"
        ]
        is False
    )
    assert set(projected[2]["series"]) == set(presentation.SERIES_FIELDS)
    for key in presentation.SERIES_FIELDS:
        assert projected[2]["series"][key] == example["series"][key]
    assert before == json.dumps((results, card, example))


def test_missing_station_evidence_stays_null():
    _, card, _ = _inputs()
    card["variables"]["temperature"].update(benchmark=None, example=None)
    projected = presentation.project_card(card)["variables"]["temperature"]
    assert projected["giano_results"] is None and projected["example"] is None


def _build_small_bundle(tmp_path, monkeypatch):
    results, card, example = _inputs()
    source = tmp_path / "source"
    payloads = {
        "results.json": results,
        "stations/T0001.json": card,
        "examples/T0001/temperature.json": example,
        "stations.geojson": {
            "features": [
                {
                    "properties": {
                        "id": "T0001",
                        "card": "stations/T0001.json",
                        "has_local_data": True,
                        "has_example": True,
                        "has_benchmark_results": True,
                    }
                }
            ]
        },
        "downstream.json": {"comparison": "never copy"},
    }
    for name, payload in payloads.items():
        presentation.write_json(source / name, payload)
    presentation.write_json(
        source / "manifest.json",
        {
            "station_catalog": {
                "source_url": "https://example.test",
                "attribution": "Provider",
            },
            "files": {name: file_sha256(source / name) for name in payloads},
            "conclusions": "legacy conclusions never copied",
        },
    )

    def fake_plot(_example, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated giano plot")

    monkeypatch.setattr(presentation, "plot_example", fake_plot)
    output = tmp_path / "output"
    presentation.build(source, output)
    return output


def test_bundle_is_built_from_allowlisted_files_not_source_archives(
    tmp_path, monkeypatch
):
    output = _build_small_bundle(tmp_path, monkeypatch)
    assert presentation.verify_bundle(output)["examples"] == 1
    assert not (output / "downstream.json").exists()
    assert not (output / "sources").exists()
    assert not (output / "README.md").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert "examples/T0001/temperature.json" in manifest["files"]
    assert all("\\" not in name for name in manifest["files"])


@pytest.mark.parametrize(
    "problem", ["extra_file", "comparison", "metric", "missing_truth"]
)
def test_verification_rejects_leaked_or_incorrect_presentation(
    tmp_path, monkeypatch, problem
):
    output = _build_small_bundle(tmp_path, monkeypatch)
    path = output / "examples/T0001/temperature.json"
    payload = json.loads(path.read_text())
    if problem == "extra_file":
        (output / "unlisted.json").write_text("{}")
    elif problem == "comparison":
        payload["metrics"]["bilstm_fair_mae"] = 1.0
    elif problem == "metric":
        payload["metrics"]["mae"] = 10.0
    else:
        payload["series"]["ground_truth"][5] = 123.0
    presentation.write_json(path, payload)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][path.relative_to(output).as_posix()] = file_sha256(path)
    presentation.write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        presentation.verify_bundle(output)


def test_relative_asset_cannot_escape_output(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        presentation.safe_path(tmp_path, "../file.json")
