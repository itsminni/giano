"""Paired transfer evaluation on Swiss NOAA stations with frozen source files."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from giano.fill import fill
from giano.physical import UNITS_BY_VARIABLE

HERE = Path(__file__).parent
ORIGINAL = ("067600-99999", "067620-99999", "067700-99999")
FIELDS = ("temperature", "wind_speed", "wind_direction")


def load(raw):
    manifest = pd.read_csv(HERE / "sources.csv", dtype={"station": str})
    records = []
    for r in manifest.itertuples():
        station = str(r.station)
        path = raw / f"{station.split('-')[0]}-{r.year}.gz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != r.sha256:
            raise ValueError(f"Unexpected source bytes for {path.name}")
        values = pd.read_csv(path, sep=r"\s+", header=None, na_values=["-9999"])
        stamp = pd.to_datetime(
            values.iloc[:, :4].set_axis(["year", "month", "day", "hour"], axis=1)
        )
        records.append(
            pd.DataFrame(
                dict(
                    station=station,
                    time=stamp,
                    temperature=values[4] / 10,
                    wind_speed=values[8] / 10,
                    wind_direction=values[7].where(values[8] > 0),
                )
            )
        )
    ds = (
        pd.concat(records)
        .set_index(["time", "station"])
        .to_xarray()
        .reindex(time=pd.date_range("2022-01-01", "2024-12-31T23", freq="h"))
    )
    coords = (
        manifest.drop_duplicates("station").set_index("station").loc[ds.station.values]
    )
    for name in ("latitude", "longitude"):
        ds = ds.assign_coords({name: ("station", coords[name].to_numpy())})
    for field in FIELDS:
        ds[field].attrs["units"] = UNITS_BY_VARIABLE[field]
    return ds


def spatial_reference(history, donors, target, direction):
    # Estimate station-to-station offsets only before this outage.
    delta = history[:, target, None] - history[:, donors]
    good = np.isfinite(delta)
    count = good.sum(axis=0)
    if direction:
        sums = np.where(good, np.exp(1j * np.deg2rad(delta)), 0).sum(axis=0)
        offset = np.rad2deg(np.angle(sums))
        offset[np.abs(sums) < 1e-6 * count] = np.nan
    else:
        offset = np.array(
            [
                np.median(d[ok]) if ok.any() else np.nan
                for d, ok in zip(delta.T, good.T, strict=True)
            ]
        )
    return np.where(count >= 24, offset, np.nan)


def evaluate_network(ds, network, out, *, local_context=False, station_reference=False):
    rows = []
    clock = ds.time.values
    for year in (2023, 2024):
        for month in (1, 4, 7, 10):
            start = np.datetime64(f"{year}-{month:02d}-15T00")
            left = clock.searchsorted(start)
            for hours in (6, 168, 720):
                right = left + hours
                sl = slice(left, right)
                for k, site in enumerate(ds.station.values):
                    donors = np.flatnonzero(ds.station.values != site)
                    masked = ds.copy(deep=True)
                    for field in FIELDS:
                        masked[field].values[sl, k] = np.nan
                    result = fill(
                        masked,
                        str(start),
                        str(clock[right]),
                        FIELDS,
                        local_context=local_context,
                        station_reference=station_reference,
                    )
                    for field in FIELDS:
                        direction = field == "wind_direction"
                        x = ds[field].values
                        offset = spatial_reference(x[:left], donors, k, direction)
                        adjusted = x[sl][:, donors] + offset
                        ix = np.flatnonzero(np.isfinite(masked[field].values[:, k]))
                        visible = x[ix, k]
                        if direction:
                            visible = np.rad2deg(np.unwrap(np.deg2rad(visible)))
                        linear = np.interp(
                            np.arange(left, right),
                            ix,
                            visible,
                            left=np.nan,
                            right=np.nan,
                        )
                        spatial = np.full(hours, np.nan)
                        available = np.isfinite(adjusted).any(axis=1)
                        if direction:
                            z = np.nansum(np.exp(1j * np.deg2rad(adjusted)), axis=1)
                            ok = available & (
                                np.abs(z) > 1e-6 * np.isfinite(adjusted).sum(axis=1)
                            )
                            spatial[ok] = np.rad2deg(np.angle(z[ok])) % 360
                        else:
                            spatial[available] = np.nanmedian(
                                adjusted[available], axis=1
                            )
                            if field == "wind_speed":
                                spatial = np.maximum(spatial, 0)
                        rows.append(
                            pd.DataFrame(
                                dict(
                                    network=network,
                                    station=site,
                                    start=start,
                                    hours=hours,
                                    field=field,
                                    time=clock[sl],
                                    truth=x[sl, k],
                                    giano=result[field + "_filled"].values[:, k],
                                    source=result[field + "_reference"].values[:, k],
                                    linear=linear,
                                    spatial=spatial,
                                )
                            )
                        )
            print(network, start, flush=True)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(out / f"hourly-{network}.csv.gz", index=False)
    return result


def summarize(sparse, dense):
    keys = ["station", "start", "hours", "field", "time"]
    a = sparse.set_index(keys)
    b = dense[dense.station.isin(ORIGINAL)].set_index(keys).loc[a.index]
    if not np.array_equal(a.truth, b.truth, equal_nan=True):
        raise ValueError("Different targets between networks")
    methods = ["giano", "linear", "spatial"]
    paired = a[["truth", *methods]].notna().all(axis=1) & b[methods].notna().all(axis=1)
    tables = []
    for label, frame, common in (
        ("3-original", a, paired),
        ("8-original", b, paired),
        ("8-all", dense.set_index(keys), None),
    ):
        frame = frame.reset_index()
        ok = (
            frame[["truth", *methods]].notna().all(axis=1)
            if common is None
            else common.to_numpy()
        )
        frame["measured"] = frame.truth.notna()
        frame["paired"] = ok
        frame["neighbors"] = (frame.source == 1) & ok
        for m in methods:
            d = frame[m] - frame.truth
            frame[m] = np.abs(
                d.where(frame.field != "wind_direction", (d + 180) % 360 - 180)
            ).where(ok)
        for dims in ([], ["station"], ["start"]):
            keys2 = ["field", "hours", *dims]
            by = frame.groupby(keys2)
            table = (
                by[methods].mean().join(by[["measured", "paired", "neighbors"]].sum())
            )
            table["scheduled"] = by.size()
            episode = (
                frame.groupby(list(dict.fromkeys([*keys2, "start", "station"])))[
                    methods
                ]
                .mean()
                .dropna()
                .reset_index()
            )
            summaries = episode.groupby(keys2)
            table["episodes"] = summaries.size()
            table["wins_vs_linear"] = (
                episode.assign(win=episode.giano < episode.linear)
                .groupby(keys2)
                .win.sum()
            )
            table["wins_vs_spatial"] = (
                episode.assign(win=episode.giano < episode.spatial)
                .groupby(keys2)
                .win.sum()
            )
            for count in ("episodes", "wins_vs_linear", "wins_vs_spatial"):
                table[count] = table[count].fillna(0).astype(int)
            table = table.reset_index().assign(network=label)
            for key in ("station", "start"):
                if key not in dims:
                    table[key] = "all"
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--local-context", action="store_true")
    p.add_argument("--station-reference", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    ds = load(args.raw)
    sparse = evaluate_network(
        ds.sel(station=list(ORIGINAL)),
        "3",
        args.output,
        local_context=args.local_context,
        station_reference=args.station_reference,
    )
    dense = evaluate_network(
        ds,
        "8",
        args.output,
        local_context=args.local_context,
        station_reference=args.station_reference,
    )
    summarize(sparse, dense).to_csv(
        args.output / "scores.csv", index=False, float_format="%.6f"
    )
