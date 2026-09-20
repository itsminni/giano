"""Version comparisons must not silently mix partial benchmark suites."""

import json

import pytest

from giano.evaluation.gapfill.compare_versions import summarize
from giano.prediction import prediction_policy


def _artifacts(root):
    checkpoint = root / "weights.pt"
    checkpoint.write_bytes(b"not loaded: provenance hash fixture")
    protocol = {
        "split": "val",
        "seeds": [42],
        "seed_mode": "paired_training_and_mask",
        "max_batches": 20,
        "cases": [{"mask_type": "block", "parameter": 24}],
        "prediction_postprocessing": prediction_policy(),
        "checkpoints": {"example": {"path": str(checkpoint)}},
    }
    for version in ("old_data", "new_data"):
        directory = root / version
        directory.mkdir()
        for family, models in (
            ("imputeformer", [("giano", "imputeformer")]),
            ("bilstm", [("bilstm", "fair"), ("bilstm", "legacy_retrained")]),
        ):
            rows = [
                {
                    "model": model,
                    "model_variant": variant,
                    "variable": "temperature",
                    "split": "val",
                    "seed": 42,
                    "mask_type": "block",
                    "mask_parameter": 24,
                    "n_hidden": 24,
                    "mae": 1.0,
                    "rmse": 2.0,
                }
                for model, variant in [*models, ("interpolation", "linear")]
            ]
            (directory / f"{family}_benchmark.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "protocol": {**protocol, "data_dir": version},
                        "results": rows,
                    }
                )
            )


def test_validation_version_summary_verifies_complete_pairing(tmp_path):
    _artifacts(tmp_path)
    result = summarize(tmp_path, variables=("temperature",))
    assert len(result["results"]) == 4
    assert len(result["checkpoint_sha256"]) == 1
    assert all(row["mae_change"] == 0 for row in result["results"])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_variable",
        "missing_model",
        "different_counts",
        "different_policy",
        "test_split",
    ],
)
def test_incomplete_or_incompatible_validation_is_rejected(tmp_path, mutation):
    _artifacts(tmp_path)
    path = tmp_path / "new_data" / "bilstm_benchmark.json"
    payload = json.loads(path.read_text())
    if mutation == "missing_variable":
        payload["results"] = []
    elif mutation == "missing_model":
        payload["results"] = payload["results"][1:]
    elif mutation == "different_counts":
        payload["results"][0]["n_hidden"] = 23
    elif mutation == "different_policy":
        payload["protocol"]["prediction_postprocessing"] = None
    else:
        payload["protocol"]["split"] = "test"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        summarize(tmp_path, variables=("temperature",))
