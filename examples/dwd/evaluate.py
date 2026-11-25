"""Compare frozen Giano modes on paired DWD station outages in 2024 and 2025."""

import argparse
import hashlib
from itertools import product
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import xarray as xr

from giano.data import make_panel
from giano.fill import Predictor
from giano.local_context import FIELDS
from giano.physical import PHYSICAL_BOUNDS_BY_VARIABLE
from giano.windows import Outage, Regime

HERE = Path(__file__).parent
PRODUCTS = {
    "pressure": {"P0": "pressure"},
    "air_temperature": {"TT_TU": "temperature", "RF_TU": "humidity"},
}
SEASONS = ("01-05T03", "04-05T09", "07-05T15", "09-30T21")
SCENARIOS: tuple[tuple[Regime, int], ...] = (
    ("other-stations", 1),
    ("other-stations", 6),
    ("other-stations", 168),
    ("other-stations", 720),
    ("other-stations", 2160),
    ("no-contemporaneous-inputs", 6),
    ("no-contemporaneous-inputs", 720),
)


def station_product(raw, station, product):
    filename = station[product]
    path = raw / filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != station[product + "_sha256"]:
        raise ValueError(f"Source digest mismatch for {filename}")
    mapping = PRODUCTS[product]
    with ZipFile(path) as archive:
        (name,) = [name for name in archive.namelist() if name.startswith("produkt_")]
        with archive.open(name) as stream:
            frame = pd.read_csv(
                stream,
                sep=";",
                skipinitialspace=True,
                na_values=["-999"],
                usecols=["STATIONS_ID", "MESS_DATUM", *mapping],
            )
    if not (frame.STATIONS_ID == int(station.site)).all():
        raise ValueError(f"Unexpected station in {filename}")
    frame = frame.loc[frame.MESS_DATUM.between(2023010100, 2025123123)].copy()
    frame.index = pd.to_datetime(frame.MESS_DATUM.astype(str), format="%Y%m%d%H")
    if not frame.index.is_unique:
        raise ValueError(f"Duplicate station hours in {filename}")
    frame = frame[list(mapping)].rename(columns=mapping)
    for key in mapping.values():
        low, high = PHYSICAL_BOUNDS_BY_VARIABLE[key]
        frame[key] = frame[key].where(frame[key].between(low, high))
    return frame


def network(raw, roster):
    records = []
    for _, station in roster.iterrows():
        readings = pd.concat(
            [station_product(raw, station, p) for p in PRODUCTS], axis=1
        )
        readings.index.name = "time"
        records.append(readings.assign(station=station.site).reset_index())
    grid = pd.date_range("2023-01-01", "2025-12-31T23", freq="h")
    ds = (
        pd.concat(records).set_index(["time", "station"]).to_xarray().reindex(time=grid)
    )
    ordered = roster.set_index("site").loc[ds.station.values]
    return ds.assign_coords(
        latitude=("station", ordered.lat.values),
        longitude=("station", ordered.lon.values),
    )


def linear_bridge(values, start, end):
    left = np.flatnonzero(np.isfinite(values[:start]))
    right = np.flatnonzero(np.isfinite(values[end:]))
    if not len(left) or not len(right):
        return np.full(end - start, np.nan)
    bounds = [left[-1], end + right[0]]
    return np.interp(np.arange(start, end), bounds, values[bounds])


def evaluate(raw, roster):
    rows = []
    for region, stations in roster.groupby("region"):
        ds = network(raw, stations)
        clock = ds.time.values
        panel = make_panel(ds, int(clock.searchsorted(np.datetime64("2024-01-01"))))
        model = Predictor(xr.Dataset(), panel, False)
        for year, season in product((2024, 2025), SEASONS):
            stamp = np.datetime64(f"{year}-{season}")
            start = int(clock.searchsorted(stamp))
            for (regime, hours), node in product(SCENARIOS, range(len(stations))):
                gap = Outage(start, start + hours, node, regime)
                base = model.reconstruct(gap)[0]
                local, methods, *_ = model.reconstruct(gap, local_context=True)
                if not np.array_equal(np.isfinite(base), np.isfinite(local)):
                    raise ValueError("Coverage changed")
                for j, field in enumerate(FIELDS):
                    series = panel.values[:, node, j].astype(float)
                    truth = series[start : start + hours]
                    linear = linear_bridge(series, start, start + hours)
                    supported = np.isfinite(truth) & np.isfinite(base[:, j])
                    common = supported & np.isfinite(linear)
                    replacement = np.isin(
                        methods[:, j],
                        [
                            "station-interpolation",
                            "offset-neighbors",
                            "calibrated-neighbors",
                        ],
                    )
                    row = dict(
                        region=region,
                        year=year,
                        start=stamp,
                        station=panel.station_ids[node],
                        regime=regime,
                        hours=hours,
                        variable=field,
                        measured=np.isfinite(truth).sum(),
                        supported=supported.sum(),
                        paired=common.sum(),
                        replaced=(replacement & common).sum(),
                    )
                    for key, est in (
                        ("standard", base[:, j]),
                        ("local", local[:, j]),
                        ("linear", linear),
                    ):
                        row[key + "_error"] = np.abs(est[common] - truth[common]).sum()
                    rows.append(row)
            print(region, stamp, flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    roster = pd.read_csv(HERE / "sources.csv", dtype={"site": str})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluate(args.raw, roster).to_csv(args.output, index=False, mode="x")
