from pathlib import Path

import numpy as np
import pytest

from giano.downstream.ridge_forecaster import complete_origins
from giano.evaluation.forecasting import run
from giano.evaluation.forecasting.run import select_station
from giano.spatiotemporal_dataset import SpatiotemporalPanel


def _panel() -> SpatiotemporalPanel:
    values = np.ones((100, 3))
    values[:75, 0] = np.nan  # perfect test coverage, unusable clean training
    values[95:, 1:] = np.nan
    station_ids = ("late", "valid", "tie")
    return SpatiotemporalPanel(
        values=values,
        times=np.datetime64("2020-01-01T00", "ns")
        + np.arange(100) * np.timedelta64(1, "h"),
        station_ids=station_ids,
        coordinates=np.array([[46.0, 11.0], [46.1, 11.1], [46.2, 11.2]]),
        auxiliary=None,
        coverage_bounds=np.array([[0, 99], [0, 99], [0, 99]]),
        source_paths=tuple(
            Path(f"{station}_humidity_merged.nc") for station in station_ids
        ),
        variable="humidity",
    )


def test_selection_requires_training_coverage_and_breaks_ties_in_panel_order():
    index, count = select_station(
        _panel(),
        history_length=4,
        horizons=(1, 2),
        stride=1,
        requested=None,
    )
    assert index == 1
    assert count == 8


def test_requested_station_with_no_training_origins_is_rejected():
    with pytest.raises(ValueError, match="complete training origins"):
        select_station(
            _panel(), history_length=4, horizons=(1, 2), stride=1, requested="late"
        )


def test_no_train_eligible_station_fails_explicitly():
    panel = _panel()
    panel.values[:75] = np.nan
    with pytest.raises(ValueError, match="both complete training and test"):
        select_station(
            panel, history_length=4, horizons=(1, 2), stride=1, requested=None
        )


def test_unavailable_campaign_variable_is_recorded_without_fake_scores(
    tmp_path, monkeypatch
):
    import json

    panel = _panel()
    panel.values[:75] = np.nan
    monkeypatch.setattr(run, "build_spatiotemporal_panel", lambda *_: panel)
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    output = tmp_path / "unavailable.json"
    assert (
        run.main(
            [
                "--config",
                str(config),
                "--variables",
                "humidity",
                "--record-unavailable",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text())
    assert payload["protocol"]["evaluation_status"] == "not_evaluable"
    assert payload["records"] == [] and payload["summaries"] == []
    assert "humidity" in payload["protocol"]["unavailable_variables"]


@pytest.mark.parametrize("start,end,stride", [(0, 70, 1), (85, 100, 2), (0, 3, 1)])
def test_complete_origins_matches_original_fitting_rule(start, end, stride):
    values = np.random.default_rng(42).normal(size=100)
    values[[3, 11, 68, 89]] = np.nan
    expected = [
        i
        for i in range(max(start, 3), end - 2, stride)
        if np.isfinite(values[i - 3 : i + 1]).all()
        and np.isfinite(values[i + np.array([1, 2])]).all()
    ]
    np.testing.assert_array_equal(
        complete_origins(
            values,
            history_length=4,
            horizons=(1, 2),
            start=start,
            end=end,
            stride=stride,
        ),
        expected,
    )
