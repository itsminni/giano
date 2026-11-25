"""Retrospective hourly gap filling on a user-supplied station network."""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
import xarray as xr

from .calibrated_neighbors import FIELDS as DONOR_FIELDS
from .calibrated_neighbors import calibrated_neighbors
from .data import ERA5_UNITS, load_input, make_panel
from .inference import _masked_panel, reconstruct_long
from .local_context import FIELDS as LOCAL_FIELDS
from .local_context import apply_local_context
from .model import VARIABLES, LongGapModel
from .physical import donor_correction, humidity_proxy
from .pressure import PressureOffset, PressureWindows
from .station_reference import apply_station_reference
from .windows import HistoryReference, Outage, fit_history

WEIGHTS = Path(__file__).parent / "weights"
HOURS = (2160, 720, 168, 6, 1)


def missing_runs(missing):
    return np.flatnonzero(np.diff(missing, prepend=False, append=False)).reshape(-1, 2)


def episodes(missing, offset, length):
    for left, right in missing_runs(missing):
        start = max(48, int(left) + offset)
        for duration in HOURS:
            context = ((duration + 71) // 72) * 72 + 48
            while start + duration <= int(right) + offset and start + context <= length:
                yield start, duration
                start += duration


def update_local_field(
    panel,
    gap,
    variable,
    prediction,
    method,
    source,
    allow_bridge,
    donor_history_end=None,
):
    updated, local_source = apply_local_context(
        panel,
        gap,
        variable,
        prediction,
        allow_bridge=allow_bridge,
        donor_history_end=donor_history_end,
    )
    prediction[:] = updated
    replaced = local_source > 0
    source[replaced] = local_source[replaced]
    method[local_source == 5] = "station-interpolation"
    method[local_source == 1] = "offset-neighbors"


def update_station_reference(
    panel, gap, variable, prediction, method, source, offset=0
):
    updated, new_source = apply_station_reference(
        panel, gap, variable, prediction, offset=offset
    )
    prediction[:] = updated
    for code, label in (
        (3, "recent-pressure-level"),
        (
            5,
            "station-interpolation"
            if variable == "pressure"
            else "circular-interpolation",
        ),
        (6, "adjacent-direction"),
    ):
        use = new_source == code
        method[use] = label
        source[use] = code


def update_donor_field(panel, gap, variable, prediction, method, source, offset=0):
    estimate = calibrated_neighbors(panel, gap, variable)[
        offset : offset + len(prediction)
    ]
    use = np.isfinite(prediction) & np.isfinite(estimate)
    prediction[use] = estimate[use]
    method[use] = "calibrated-neighbors"
    source[use] = 1


class Predictor:
    def __init__(self, ds, panel, trentino, ensemble=None):
        self.panel = panel
        self.ds = ds
        product = json.loads((WEIGHTS / "product.json").read_text())
        for name, expected in product["sha256"].items():
            if hashlib.sha256((WEIGHTS / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Changed model asset: {name}")
        self.models = []
        path = ensemble or WEIGHTS / "ensemble.pt"
        self.ensemble_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        states = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(states, list) or len(states) != 2:
            raise ValueError(
                "Expected the two state dictionaries exported by giano-train"
            )
        for state in states:
            model = LongGapModel()
            model.load_state_dict(state)
            self.models.append(model.eval())
        self.calibration = json.loads((WEIGHTS / "calibration.json").read_text())
        self.surface = (
            ds["sp"].values.astype(float) / 100
            if "sp" in ds
            else np.full(panel.values.shape[:2], np.nan)
        )
        self.trentino = trentino
        if trentino:
            with np.load(WEIGHTS / "trentino.npz", allow_pickle=False) as stored:
                roster = stored["stations"].tolist()
                nodes = [roster.index(s) for s in panel.station_ids]
                if not np.allclose(
                    panel.coordinates, stored["coordinates"][nodes], atol=1e-4, rtol=0
                ):
                    raise ValueError(
                        "Trentino IDs and coordinates must match the retained calibration"
                    )
                self.history = HistoryReference(
                    stored["center"][nodes],
                    stored["scale"][nodes],
                    stored["monthly"][:, nodes],
                    stored["counts"][nodes],
                    product["sha256"]["trentino.npz"],
                )
                self.offset = PressureOffset(
                    stored["offset"][nodes],
                    stored["pressure_count"][nodes],
                    stored["pressure_valid"][nodes],
                )
        else:
            humidity_mask = (
                ds["humidity_reference_valid"].transpose("time", "station").values
                if "humidity_reference_valid" in ds
                else None
            )
            self.history = fit_history(panel, humidity_reference_valid=humidity_mask)
            observed = panel.values[: panel.train_end, :, 2]
            surface = self.surface[: panel.train_end]
            pairs = np.isfinite(observed) & np.isfinite(surface)
            count = pairs.sum(axis=0)
            offsets = np.divide(
                np.where(pairs, observed - surface, 0).sum(axis=0),
                count,
                out=np.zeros(len(count)),
                where=count >= 24,
            )
            self.offset = PressureOffset(offsets, count, count >= 24)

    def predict(
        self,
        node,
        start,
        hours,
        *,
        local_context=False,
        allow_bridge=True,
        station_reference=False,
    ):
        panel = self.panel
        sl = slice(start, start + hours)
        visible = np.isfinite(panel.values[sl]) & panel.availability[sl]
        visible[:, node] = False
        regime = "other-stations" if visible.any() else "external-only"
        gap = Outage(start, start + hours, node, regime)
        prediction, method, source, total, _, _ = self.reconstruct(
            gap,
            local_context=local_context,
            allow_bridge=allow_bridge,
            station_reference=station_reference,
        )
        return prediction, method, source, total

    def reconstruct(
        self, gap, *, local_context=False, allow_bridge=True, station_reference=False
    ):
        panel, node, regime = self.panel, gap.node, gap.regime
        sl = slice(gap.start, gap.end)
        hours = gap.end - gap.start
        windows = PressureWindows(
            _masked_panel(panel, gap), gap, _evaluation_history=lambda: self.history
        )
        windows.surface, windows.offset = self.surface, self.offset
        a, b = (reconstruct_long(model, windows) for model in self.models)
        point = (a.values + b.values) / 2
        prediction = point.copy()
        method = np.full(prediction.shape, "point-pair", dtype="U24")
        method[:, 4:] = "physical-wind"
        station = panel.station_ids[node]
        total = None
        if hours in (720, 2160):
            if regime == "other-stations":
                for i in range(len(windows)):
                    core = slice(i * 72, min((i + 1) * 72, hours))
                    prediction[core, 3], used = donor_correction(
                        windows[i], prediction[core, 3], self.calibration["rain"]
                    )
                    method[core, 3] = np.where(used, "rain-neighbors", "point-pair")
            elif self.trentino and regime == "external-only":
                coef = self.calibration["humidity"].get(station)
                if coef is not None and all(k in self.ds for k in ERA5_UNITS):
                    fields = {k: self.ds[k].values[sl, node] for k in ERA5_UNITS}
                    proxy = humidity_proxy(panel.times[sl], fields, coef)
                    use = np.isfinite(proxy) & np.isfinite(prediction[:, 1])
                    prediction[use, 1] = (prediction[use, 1] + proxy[use]) / 2
                    method[use, 1] = "humidity-era5"
                rain = panel.auxiliary[sl, node, 3]
                ratio = self.calibration["totals"].get(station)
                if (
                    ratio is not None
                    and (np.isfinite(rain) & panel.availability[sl, node, 3]).all()
                ):
                    total = float(np.clip(rain, 0, 300).sum(dtype=np.float64) * ratio)
        # On new stations an unsupported population prior alone cannot justify a fill.
        support = (a.baseline_source != 3) | (self.history.counts[node] >= 4)
        if not self.trentino:
            prediction[~support] = np.nan
            method[~support] = "no-local-reference"
        method[~np.isfinite(prediction) & (method != "no-local-reference")] = (
            "unavailable"
        )
        source = a.baseline_source
        if local_context:
            for variable in LOCAL_FIELDS:
                col = VARIABLES.index(variable)
                update_local_field(
                    panel,
                    gap,
                    variable,
                    prediction[:, col],
                    method[:, col],
                    source[:, col],
                    allow_bridge,
                )
            if hours >= 168:
                for variable in DONOR_FIELDS:
                    col = VARIABLES.index(variable)
                    update_donor_field(
                        panel,
                        gap,
                        variable,
                        prediction[:, col],
                        method[:, col],
                        source[:, col],
                    )
        if station_reference:
            for col in (2, 5):
                update_station_reference(
                    panel,
                    gap,
                    VARIABLES[col],
                    prediction[:, col],
                    method[:, col],
                    source[:, col],
                )
        return prediction, method, source, total, point, windows


def hour(text):
    t = pd.Timestamp(text)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    if pd.isna(t) or t != t.floor("h"):
        raise ValueError("Use a whole-hour timestamp")
    return t.to_datetime64().astype("datetime64[ns]")


def fill(
    ds: xr.Dataset,
    start: str,
    end: str,
    variables=None,
    trentino=False,
    contemporary_only=False,
    ensemble: Path | None = None,
    *,
    local_context: bool = False,
    station_reference: bool = False,
) -> xr.Dataset:
    first, last = hour(start), hour(end)
    times = ds.time.values
    if not times[0] <= first < last <= times[-1] + np.timedelta64(1, "h"):
        raise ValueError("Requested period outside input time coverage")
    lower, upper = map(int, times.searchsorted([first, last]))
    cols = [
        VARIABLES.index(v) for v in (variables or [v for v in VARIABLES if v in ds])
    ]
    if any(VARIABLES[j] not in ds for j in cols):
        raise ValueError("Requested variable absent from input")
    panel = make_panel(ds, lower)
    predictor = Predictor(ds, panel, trentino, ensemble)
    result = ds.isel(time=slice(lower, upper)).copy()
    totals = []
    reference_codes = "0=none, 1=neighbor stations, 2=external, 3=history"
    if local_context:
        reference_codes += ", 5=station endpoints"
    if station_reference:
        reference_codes += ", 6=single station endpoint"
        if not local_context:
            reference_codes += ", 5=station endpoints"
    for col in cols:
        var = VARIABLES[col]
        observed = result[var].values
        status = np.where(np.isfinite(observed), "observed", "outside-record").astype(
            "U24"
        )
        status[~np.isfinite(observed) & panel.availability[lower:upper, :, col]] = (
            "insufficient-context"
        )
        result[var + "_filled"] = (
            ("time", "station"),
            observed.astype(float).copy(),
            dict(ds[var].attrs),
        )
        result[var + "_method"] = (("time", "station"), status)
        result[var + "_reference"] = (
            ("time", "station"),
            np.zeros(observed.shape, np.uint8),
            {"codes": reference_codes},
        )
    for node, station in enumerate(panel.station_ids):
        runs = defaultdict(list)
        for col in cols:
            missing = panel.availability[lower:upper, node, col] & ~np.isfinite(
                panel.values[lower:upper, node, col]
            )
            for episode in episodes(missing, lower, len(times)):
                runs[episode].append(col)
        for (t, hours), columns in runs.items():
            prediction, methods, source, total = predictor.predict(node, t, hours)
            rows = slice(t - lower, t - lower + hours)
            for col in columns:
                var = VARIABLES[col]
                result[var + "_filled"].values[rows, node] = prediction[:, col]
                result[var + "_method"].values[rows, node] = methods[:, col]
                result[var + "_reference"].values[rows, node] = source[:, col]
            if total is not None and 3 in columns:
                totals.append(
                    dict(
                        station=station,
                        start=str(times[t]),
                        hours=hours,
                        total_mm=total,
                    )
                )
        for col in cols:
            var = VARIABLES[col]
            filled = result[var + "_filled"].values[:, node]
            method = result[var + "_method"].values[:, node]
            source = result[var + "_reference"].values[:, node]
            local = local_context and var in LOCAL_FIELDS
            station_level = (
                station_reference and not contemporary_only and col in (2, 5)
            )
            donors = local_context and var in DONOR_FIELDS
            if local or station_level or donors:
                # Use the true instrument outage even when only part of it was requested.
                missing = panel.availability[:, node, col] & ~np.isfinite(
                    panel.values[:, node, col]
                )
                for onset_index, end_index in missing_runs(missing):
                    onset, stop = int(onset_index), int(end_index)
                    left, right = max(lower, onset), min(upper, stop)
                    if left >= right:
                        continue
                    rows = slice(left - lower, right - lower)
                    full_gap = Outage(onset, stop, node, "other-stations")
                    if local:
                        update_local_field(
                            panel,
                            Outage(left, right, node, "other-stations"),
                            var,
                            filled[rows],
                            method[rows],
                            source[rows],
                            not contemporary_only,
                            onset if onset < left else None,
                        )
                    if station_level:
                        update_station_reference(
                            panel,
                            full_gap,
                            var,
                            filled[rows],
                            method[rows],
                            source[rows],
                            left - onset,
                        )
                    if donors and stop - onset >= 168:
                        update_donor_field(
                            panel,
                            full_gap,
                            var,
                            filled[rows],
                            method[rows],
                            source[rows],
                            left - onset,
                        )
            if contemporary_only:
                withheld = (
                    (source == 3) & np.isfinite(filled) & (method != "humidity-era5")
                )
                filled[withheld] = np.nan
                method[withheld] = "historical-reference"
    if 5 in cols and "wind_speed" in result:
        calm = (result.wind_speed.values == 0) & ~np.isfinite(
            result.wind_direction.values
        )
        result.wind_direction_filled.values[calm] = np.nan
        result.wind_direction_method.values[calm] = "calm-wind"
        result.wind_direction_reference.values[calm] = 0
    result.attrs.update(
        model="Giano",
        ensemble_sha256=predictor.ensemble_sha256,
        reference_sha256=predictor.history.sha256,
        reference_mode="trentino" if trentino else "input history before start",
        require_contemporary_reference=int(contemporary_only),
        independent_rain_totals=json.dumps(totals),
    )
    if local_context:
        result.attrs["local_context"] = 1
    if station_reference:
        result.attrs["station_reference"] = 1
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--stations", type=Path, help="CSV of station IDs and coordinates for CSV input"
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--variables", choices=VARIABLES, nargs="+")
    parser.add_argument(
        "--trentino-calibration",
        action="store_true",
        help="Use retained references for matching historical station IDs and coordinates",
    )
    parser.add_argument(
        "--contemporary-only",
        action="store_true",
        help="Withhold estimates based on a historical physical reference alone",
    )
    parser.add_argument(
        "--ensemble", type=Path, help="Use the ensemble.pt exported by giano-train"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--local-context",
        action="store_true",
        help="Use station endpoints and calibrated neighbouring measurements where supported",
    )
    parser.add_argument(
        "--station-reference",
        action="store_true",
        help="Check pressure against its recent station level and use adjacent directions for gaps up to 12 hours",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output .nc path")
    torch.set_num_threads(2)
    ds = load_input(args.input, args.stations)
    result = fill(
        ds,
        args.start,
        args.end_exclusive,
        args.variables,
        args.trentino_calibration,
        args.contemporary_only,
        args.ensemble,
        local_context=args.local_context,
        station_reference=args.station_reference,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        staged = Path(tmp) / "output.nc"
        result.to_netcdf(staged, engine="h5netcdf")
        os.link(staged, args.output)
    for var in VARIABLES:
        if var + "_method" in result:
            keys, counts = np.unique(result[var + "_method"].values, return_counts=True)
            print(var, dict(zip(keys.tolist(), counts.tolist(), strict=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
