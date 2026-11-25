from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
import xarray as xr
from ridge.evaluate import masked_batch, model_inputs, past_window
from ridge.evaluate_context import context_features, repairs_with_prehistory

from giano.data import make_panel
from giano.inference import _ambiguous_direction
from giano.model import INPUTS
from giano.pressure import PressureOffset
from giano.windows import Outage, fit_history


def test_extended_context_is_origin_bounded_and_independent_of_repaired_lags():
    times = np.datetime64("2020-01-01") + np.arange(780) * np.timedelta64(1, "h")
    pressure = np.tile(np.arange(780, dtype=float)[:, None], (1, 4))
    ds = xr.Dataset(
        {"pressure": (("time", "station"), pressure)},
        coords={
            "time": times,
            "station": list("ABCD"),
            "latitude": ("station", [46, 46.1, 46.2, 46.3]),
            "longitude": ("station", [11, 11, 11, 11]),
        },
    )
    panel = make_panel(ds, 720)
    end = 750
    original = context_features(panel, 0, "pressure")[end]
    np.testing.assert_allclose(
        original[:2], [(end - 167 + end - 72) / 2, (end - 719 + end - 168) / 2]
    )
    panel.values[end - 71 : end + 1, 0] = -1000
    panel.values[end + 1 :] = -1000
    np.testing.assert_equal(context_features(panel, 0, "pressure")[end], original)


def test_repairers_share_the_earlier_observation_and_stop_at_origin():
    times = np.datetime64("2020-01-01") + np.arange(250) * np.timedelta64(1, "h")
    ds = xr.Dataset(
        {"pressure": (("time", "station"), 800.0 + np.arange(250)[:, None] / 10)},
        coords={
            "time": times,
            "station": ["A"],
            "latitude": ("station", [46]),
            "longitude": ("station", [11]),
        },
    )
    p = make_panel(ds, 0)
    masks = np.zeros((1, 72), bool)
    masks[:, :12] = True
    point = np.full((1, 72), 805.0)
    linear, local = repairs_with_prehistory(p, 240, 0, "pressure", masks, point)
    for repaired in (linear, local):
        np.testing.assert_allclose(repaired[0], p.values[169:241, 0, 2], atol=1e-4)
    p.values[169:181, 0, 2] = 1050
    p.values[241:, 0, 2] = 1050
    after = repairs_with_prehistory(p, 240, 0, "pressure", masks, point)
    np.testing.assert_equal(after, (linear, local))


def test_forecast_inputs_exclude_hidden_readings_and_every_hour_after_origin():
    rng = np.random.default_rng(42)
    times = np.datetime64("2020-01-01T00") + np.arange(300) * np.timedelta64(1, "h")
    obs = rng.normal(10, 2, (300, 5))
    ds = xr.Dataset(
        {"temperature": (("time", "station"), obs)},
        coords={
            "time": times,
            "station": list("ABCDE"),
            "latitude": ("station", np.arange(45, 50)),
            "longitude": ("station", np.zeros(5)),
        },
    )
    panel = make_panel(ds, 80)
    predictor = SimpleNamespace(
        history=fit_history(panel),
        offset=PressureOffset(np.zeros(5), np.zeros(5), np.zeros(5, bool)),
    )
    node, end = 0, 220
    mask = np.arange(72) % 3 == 0
    original = past_window(panel, end)
    gap = Outage(48, 120, node, "other-stations")
    surface = np.full((72, 5), np.nan)
    direct = model_inputs(original, gap, predictor, surface, mask)
    changed = replace(
        panel,
        values=panel.values.copy(),
        auxiliary=panel.auxiliary.copy(),
        availability=panel.availability.copy(),
    )
    changed.values[end - 71 + np.flatnonzero(mask), node, 0] += 100
    changed.values[end + 1 :] = 200
    changed.auxiliary[end + 1 :] = 100
    changed.availability[end + 1 :] = False
    perturbed = model_inputs(past_window(changed, end), gap, predictor, surface, mask)
    visible = model_inputs(original, gap, predictor, surface, np.zeros(72, bool))
    hidden = model_inputs(original, gap, predictor, surface, np.ones(72, bool))
    batch = masked_batch(visible, hidden, mask[None])
    assert direct["calendar"].shape[0] == 72
    assert direct["input_mask"][:, 0, 0].tolist() == (~mask).tolist()
    for key in INPUTS:
        assert torch.equal(direct[key], perturbed[key])
        assert torch.equal(direct[key], batch[key][0])


def test_wind_history_reference_uses_only_prior_calibration_readings():
    times = np.datetime64("2020-01-01T00") + np.arange(144) * np.timedelta64(1, "h")
    ds = xr.Dataset(
        {"wind_direction": (("time", "station"), np.zeros((144, 1)))},
        coords={
            "time": times,
            "station": ["A"],
            "latitude": ("station", [46.0]),
            "longitude": ("station", [11.0]),
        },
    )
    calibration = make_panel(ds, 48)
    calibration.values[1:48:2, 0, 5] = 180
    current = make_panel(ds.isel(time=slice(48, None)), 0)
    gap = Outage(0, 24, 0, "no-contemporaneous-inputs")
    sources = np.full(24, 3)
    assert not _ambiguous_direction(current, gap, sources).any()
    assert _ambiguous_direction(current, gap, sources, history_panel=calibration).all()
    calibration.values[48:, 0, 5] = 90
    assert _ambiguous_direction(current, gap, sources, history_panel=calibration).all()
    calibration.values[:48, 0, 5] = 90
    assert not _ambiguous_direction(
        current, gap, sources, history_panel=calibration
    ).any()
