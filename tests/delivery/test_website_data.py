"""Static assets must not invent station positions, measurements or improvements."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from giano.spatiotemporal_dataset import SpatiotemporalPanel

spec = importlib.util.spec_from_file_location(
    "website_data",
    Path(__file__).resolve().parents[2] / "tools/prepare_website_data.py",
)
assert spec is not None and spec.loader is not None
website = importlib.util.module_from_spec(spec)
spec.loader.exec_module(website)


def _catalog():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "properties": {
                    "codice": "T0001",
                    "nome": "Historic station",
                    "quota": "1000",
                    "inizio": "1950-01-01",
                    "fine": "2010-01-01",
                },
                "geometry": {"type": "Point", "coordinates": [11.1, 46.2]},
            }
        ],
    }


def test_catalog_retains_historical_and_uses_longitude_first():
    result = website.normalize_catalog(_catalog())["T0001"]
    assert result["geometry"]["coordinates"] == [11.1, 46.2]
    assert result["provider_status"] == "historical"
    assert result["variables"] == {}


@pytest.mark.parametrize("problem", ["duplicate", "coordinates", "unsafe_id"])
def test_catalog_rejects_invalid_station_identifiers_and_positions(problem):
    catalog = _catalog()
    if problem == "duplicate":
        catalog["features"] *= 2
    elif problem == "coordinates":
        catalog["features"][0]["geometry"]["coordinates"] = [11, 900]
    else:
        catalog["features"][0]["properties"]["codice"] = "../file"
    with pytest.raises(ValueError):
        website.normalize_catalog(catalog)


def test_percentage_is_signed_and_zero_reference_is_null():
    assert website.improvement(2, 4) == 50
    assert website.improvement(6, 4) == -50
    assert website.improvement(0, 0) is None


def test_station_aggregation_does_not_hide_incomplete_coverage():
    row = {"seed": 42, "mask_type": "block", "mask_parameter": 12, "n_hidden": 12}
    for method in ("giano", "bilstm_fair", "interpolation"):
        row[f"{method}_mae"] = 1.0
        row[f"{method}_rmse"] = 2.0
    result = website.station_scores([row])
    assert result["available_case_seed_groups"] == 1
    assert result["complete_case_seed_coverage"] is False
    with pytest.raises(ValueError, match="Duplicate"):
        website.station_scores([row, row])


class Baseline(torch.nn.Module):
    def forward(self, features, coordinates, node_mask, baseline):
        return baseline, torch.zeros_like(baseline)


def _example_dataset():
    length = 1000
    values = np.tile(np.sin(np.arange(length) / 12)[:, None], (1, 4)).astype(np.float32)
    values[720:724, 3] = np.nan
    panel = SpatiotemporalPanel(
        variable="temperature",
        times=np.arange(length).astype("datetime64[h]"),
        station_ids=("T0001", "T0002", "T0003", "T0004"),
        coordinates=np.tile([46.0, 11.0], (4, 1)),
        values=values,
        auxiliary=None,
        coverage_bounds=np.tile([0, length - 1], (4, 1)),
        source_paths=(),
    )
    return website.StationExampleDataset(
        panel,
        "val",
        max_nodes=2,
        seq_len=72,
        stride=72,
        mask_mode="block",
        block_lengths=(12,),
    )


def test_example_includes_clicked_station_and_preserves_visibility_and_missing_truth():
    dataset = _example_dataset()
    result = website.example_payload(Baseline(), dataset, 3)
    assert result is not None
    assert result["station"] == "T0004"
    assert result["illustration_only"] and not result["included_in_headline_scores"]
    assert result == website.example_payload(Baseline(), dataset, 3)
    series = result["series"]
    for target, visible, prediction, hidden, unknown in zip(
        series["ground_truth"],
        series["observed_with_mask"],
        series["reconstruction"],
        series["synthetic_mask"],
        series["ground_truth_unavailable"],
        strict=True,
    ):
        if unknown:
            assert target is None and visible is None and prediction is None
            assert not hidden
        elif hidden:
            assert target is not None and visible is None and prediction is not None
        else:
            assert target == visible == prediction


def test_example_returns_null_if_station_has_no_validation_data():
    dataset = _example_dataset()
    dataset.panel.values[700:850, 3] = np.nan
    assert website.example_payload(Baseline(), dataset, 3) is None


def test_json_writer_rejects_nonfinite_values(tmp_path):
    with pytest.raises(ValueError):
        website.write_json(tmp_path / "broken.json", {"metric": float("nan")})
