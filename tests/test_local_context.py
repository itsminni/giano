from dataclasses import replace

import numpy as np
import torch
import xarray as xr

from giano.data import make_panel
from giano.fill import fill
from giano.local_context import apply_local_context
from giano.station_reference import apply_station_reference, station_reference
from giano.windows import Outage, nearest_nodes


def network():
    times = np.datetime64("2020-01-01T00") + np.arange(360) * np.timedelta64(1, "h")
    varying = np.sin(np.arange(len(times)) / 8)
    values = varying[:, None] + np.array([[10.0, 15.0, 20.0]])
    return xr.Dataset(
        {"temperature": (("time", "station"), values, {"units": "degC"})},
        coords={
            "time": times,
            "station": ["A", "B", "C"],
            "latitude": ("station", [46.0, 46.1, 46.2]),
            "longitude": ("station", [10.0, 10.0, 10.0]),
        },
    )


def test_multiblock_forecast_never_reads_hidden_targets_or_right_padding():
    panel = make_panel(network(), 0)
    # Two terminal-history masks separated by a visible hour.
    hidden = np.zeros(72, bool)
    hidden[30:32] = True
    hidden[48:] = True
    gap = Outage(80, 152, 0, "other-stations")
    visible = panel.values.copy()
    visible[gap.end :] = np.nan
    panel = replace(panel, values=visible)
    base = np.full(72, 20.0)
    base[70] = np.nan
    updated, code = apply_local_context(panel, gap, "temperature", base, hidden=hidden)
    assert np.all(code[30:32] == 5)
    assert np.all(code[48:70] == 1)
    assert code[70] == 0 and np.isnan(updated[70])
    np.testing.assert_allclose(updated[48:70], visible[128:150, 0, 0], atol=2e-6)
    perturbed = visible.copy()
    perturbed[gap.start + np.flatnonzero(hidden), 0, 0] = 50.0
    again, ids = apply_local_context(
        replace(panel, values=perturbed), gap, "temperature", base, hidden=hidden
    )
    np.testing.assert_equal(again, updated)
    np.testing.assert_equal(ids, code)
    _, none = apply_local_context(
        panel, replace(gap, regime="external-only"), "temperature", base, hidden=hidden
    )
    assert not none[48:].any()


def test_file_filling_uses_real_endpoints_across_episode_boundaries_and_labels_donors():
    torch.set_num_threads(2)
    data = network()
    data.temperature.values[128:133, 0] = np.nan
    for contemporary, method, code in [
        (False, "station-interpolation", 5),
        (True, "offset-neighbors", 1),
    ]:
        result = fill(
            data,
            str(data.time.values[125]),
            str(data.time.values[140]),
            ["temperature"],
            local_context=True,
            contemporary_only=contemporary,
        )
        missing = np.isnan(result.temperature.values[:, 0])
        assert missing.sum() == 5
        assert (result.temperature_method.values[missing, 0] == method).all()
        assert (result.temperature_reference.values[missing, 0] == code).all()
        if not contemporary:
            expected = np.linspace(
                data.temperature.values[127, 0], data.temperature.values[133, 0], 7
            )[1:-1]
            np.testing.assert_allclose(
                result.temperature_filled.values[missing, 0], expected, atol=2e-6
            )
        np.testing.assert_equal(
            result.temperature_filled.values[~missing, 0],
            result.temperature.values[~missing, 0],
        )


def test_donor_needs_24_pre_gap_pairs_and_cannot_restore_unsupported_output():
    panel = make_panel(network(), 0)
    gap = Outage(180, 190, 0, "other-stations")
    panel.values[:157, 0, 0] = np.nan
    base = np.full(10, -10.0)
    _, source = apply_local_context(panel, gap, "temperature", base)
    assert (source == 0).all()
    panel.values[156, 0, 0] = 10
    base[0] = np.nan
    adjusted, source = apply_local_context(panel, gap, "temperature", base)
    assert source.tolist() == [0, *([1] * 9)]
    assert np.isnan(adjusted[0])


def test_file_gap_reference_survives_block_and_request_boundaries():
    torch.set_num_threads(2)
    data = network()
    # A 175-hour gap splits into 168, 6 and 1. B returns after a two-hour outage.
    data.temperature.values[55:230, 0] = np.nan
    data.temperature.values[223:225, 1:] = np.nan
    for start in (55, 222):
        result = fill(
            data,
            str(data.time.values[start]),
            str(data.time.values[231]),
            contemporary_only=True,
            local_context=True,
        ).sel(station="A")
        comparison = result.sel(time=data.time.values[222:230])
        assert comparison.temperature_method.values.tolist() == [
            "offset-neighbors",
            "historical-reference",
            "historical-reference",
            *(["offset-neighbors"] * 5),
        ]
        expected = data.temperature.sel(station="B").values[222:230] - 5
        np.testing.assert_allclose(comparison.temperature_filled, expected, atol=2e-6)
        np.testing.assert_equal(
            comparison.temperature_reference.values, [1, 3, 3, 1, 1, 1, 1, 1]
        )
        assert result.temperature_method.values[-1] == "observed"


def test_nearest_stations_across_the_antimeridian():
    panel = replace(
        make_panel(network(), 0),
        coordinates=np.array([[0.0, 179.9], [0.0, -179.9], [0.0, 175.0]]),
    )
    np.testing.assert_equal(nearest_nodes(panel, 0, 2), [0, 1])
    np.testing.assert_equal(nearest_nodes(panel, 1, 2), [1, 0])


def test_pressure_checks_support_and_recent_level():
    base = np.array([801.0, 927.0, np.nan])
    for n in (23, 24):
        checked, sources = station_reference(
            "pressure", np.full(n, 800.0), 950, base, 3
        )
        np.testing.assert_equal(checked, [801, 800 if n == 24 else 927, np.nan])
        np.testing.assert_equal(sources, [0, 3 if n == 24 else 0, 0])


def test_pressure_bridge_uses_true_gap_boundaries_and_only_finite_predictions():
    baseline = np.array([930.0, np.nan])
    checked, source = station_reference(
        "pressure", np.array([800.0]), 813.0, baseline, 12, 3
    )
    np.testing.assert_equal(checked, [804.0, np.nan])
    np.testing.assert_equal(source, [5, 0])
    for after, length in ((np.nan, 12), (813.0, 13)):
        checked, source = station_reference(
            "pressure", np.array([800.0]), after, baseline, length
        )
        np.testing.assert_equal(checked, baseline)
        np.testing.assert_equal(source, [0, 0])
    ds = network().rename(temperature="pressure")
    ds.pressure.values[:] += 800
    ds.pressure.attrs["units"] = "hPa"
    ds.pressure.values[123:135, 0] = np.nan
    ds.pressure.values[[122, 135], 0] = [800, 813]
    out = fill(
        ds, str(ds.time.values[126]), str(ds.time.values[130]), station_reference=True
    )
    np.testing.assert_allclose(out.pressure_filled.values[:, 0], [804, 805, 806, 807])
    assert (out.pressure_method.values[:, 0] == "station-interpolation").all()


def test_wind_uses_true_gap_and_preserves_unsupported_hours():
    base = np.array([100.0, np.nan, 102.0])
    values, codes = station_reference(
        "wind_direction", np.array([350.0]), 10.0, base, 3
    )
    np.testing.assert_equal(values, [355.0, np.nan, 5.0])
    np.testing.assert_equal(codes, [5, 0, 5])
    for after, duration in [(170.0, 3), (10.0, 13)]:
        values, _ = station_reference(
            "wind_direction", np.array([350.0]), after, base, duration
        )
        np.testing.assert_equal(values, base)


def test_direction_history_never_sees_hidden_values_or_future_at_origin():
    panel = make_panel(network(), 0)
    panel.availability[:, 0, 5] = True
    panel.values[:, 0, 5] = 300.0
    gap = Outage(80, 152, 0, "other-stations")
    hidden = np.zeros(72, bool)
    hidden[-12:] = True
    panel.values[139, 0, 5] = 37
    values, codes = apply_station_reference(
        panel, gap, "wind_direction", np.full(72, 100.0), hidden=hidden
    )
    np.testing.assert_equal(values[-12:], np.full(12, 37.0))
    np.testing.assert_equal(codes[-12:], np.full(12, 6))


def test_file_output_sliced_inside_wind_gap():
    torch.set_num_threads(2)
    ds = network().rename(temperature="wind_direction")
    ds.wind_direction.attrs["units"] = "degree"
    ds.wind_direction.values[124:132, 0] = np.nan
    ds.wind_direction.values[123, 0] = 350.0
    ds.wind_direction.values[132, 0] = 8.0
    out = fill(
        ds, str(ds.time.values[126]), str(ds.time.values[132]), station_reference=True
    )
    np.testing.assert_equal(
        out.wind_direction_filled.values[:, 0], [356.0, 358.0, 0.0, 2.0, 4.0, 6.0]
    )
    assert (out.wind_direction_method.values[:, 0] == "circular-interpolation").all()


def test_long_wind_gap_uses_only_prior_pairs_and_preserves_missing_support():
    from giano.calibrated_neighbors import calibrated_neighbors

    rng = np.random.default_rng(70)
    times = np.datetime64("2023-01-01") + np.arange(1200) * np.timedelta64(1, "h")
    donor = rng.uniform(0, 360, len(times))
    values = np.column_stack(((donor + 35) % 360, donor))
    ds = xr.Dataset(
        {"wind_direction": (("time", "station"), values, {"units": "degree"})},
        coords={
            "time": times,
            "station": ["A", "B"],
            "latitude": ("station", [46, 46.1]),
            "longitude": ("station", [11, 11.1]),
        },
    )
    start, end = 800, 980
    # When no donor reports, only the previous Giano estimate may be retained.
    ds.wind_direction.values[start + 3, 1] = np.nan
    panel = make_panel(ds, 0)
    gap = Outage(start, end, 0, "other-stations")
    predicted = calibrated_neighbors(panel, gap, "wind_direction")
    assert np.isnan(predicted[3])
    error = (predicted - values[start:end, 0] + 180) % 360 - 180
    assert np.nanmax(abs(error)) < 1
    panel.values[start:, 0, 5] = 123
    panel.values[end:, 1, 5] = 99
    np.testing.assert_equal(
        calibrated_neighbors(panel, gap, "wind_direction"), predicted
    )
    assert np.isnan(
        calibrated_neighbors(
            panel, replace(gap, regime="external-only"), "wind_direction"
        )
    ).all()
    panel.values[:100, 0, 5] = np.nan
    assert np.isnan(calibrated_neighbors(panel, gap, "wind_direction")).all()
    ds.wind_direction.values[start:end, 0] = np.nan
    torch.set_num_threads(2)
    for left in (start, start + 60):
        out = fill(ds, str(times[left]), str(times[left + 8]), local_context=True)
        np.testing.assert_allclose(
            out.wind_direction_filled.values[:, 0][5:],
            predicted[left - start + 5 : left - start + 8],
            atol=1e-4,
        )
        assert (
            out.wind_direction_method.values[:, 0][5:] == "calibrated-neighbors"
        ).all()
        assert (out.wind_direction_reference.values[:, 0][5:] == 1).all()


def test_packaged_humidity_calibration_uses_qualified_training_support():
    from giano.fill import WEIGHTS

    with np.load(WEIGHTS / "trentino.npz") as calibration:
        sites = list(calibration["stations"])
        empty, measured = sites.index("T0153"), sites.index("T0010")
        assert calibration["counts"][empty, 1] == 0
        assert calibration["counts"][measured, 1] == 1528
        assert np.isclose(calibration["center"][measured, 1], 68.525)
        # The zero-support station inherits monthly regional humidity.
        assert 70 < calibration["monthly"][8, empty, 1] < 85


def test_humidity_reference_mask_preserves_the_original_measurements():
    from giano.fill import Predictor

    ds = network().rename(temperature="humidity")
    ds["humidity"].values[:50, 0] = 0.4
    ds["humidity"].values[50:, 0] = 80.0
    mask = np.ones(ds.humidity.shape, dtype=np.uint8)
    mask[:50, 0] = 0
    ds["humidity_reference_valid"] = (("time", "station"), mask)
    panel = make_panel(ds, 70)
    history = Predictor(ds, panel, False).history
    assert history.counts[0, 1] == 20
    assert history.center[0, 1] == 80
    np.testing.assert_allclose(panel.values[:50, 0, 1], 0.4)
