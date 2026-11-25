"""Measure the effect of DWD repairs on a common origin-bounded Ridge forecast."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from ridge.evaluate import past_window, repair
from ridge.learning_curve_fit import HORIZONS, calendar, pooled_fit

from giano.data import make_panel
from giano.fill import Predictor
from giano.local_context import FIELDS

from .evaluate import HERE, network


def fit_ridge(panel, col):
    parts = []
    origins = np.arange(71, panel.train_end - max(HORIZONS))
    for node in range(len(panel.station_ids)):
        y = panel.values[: panel.train_end, node, col].astype(float)
        windows = np.lib.stride_tricks.sliding_window_view(y, 72)
        xy = np.column_stack(
            (
                windows[origins - 71],
                calendar(panel.times[origins]),
                y[origins[:, None] + HORIZONS],
            )
        )
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy):
            center = xy.mean(axis=0)
            xy -= center
            parts.append((len(xy), center, xy.T @ xy))
    return pooled_fit(parts)


def evaluate_forecasts(raw, roster):
    rows = []
    lag = np.arange(72)
    for region, stations in roster.groupby("region"):
        ds = network(raw, stations)
        clock = ds.time.values
        panel = make_panel(ds, int(clock.searchsorted(np.datetime64("2024-01-01"))))
        model = Predictor(xr.Dataset(), panel, False)
        for col, variable in enumerate(FIELDS):
            beta, intercept, n = fit_ridge(panel, col)
            for year, (month, hour), day in product(
                (2024, 2025), ((2, 3), (5, 9), (8, 15), (11, 21)), (10, 24)
            ):
                stamp = np.datetime64(f"{year}-{month:02d}-{day:02d}T{hour:02d}")
                origin = clock.searchsorted(stamp)
                sample = past_window(panel, origin)
                rng = np.random.default_rng(
                    42 + int(stamp.astype("datetime64[h]").astype(int))
                )
                point = np.zeros(72, bool)
                point[rng.choice(72, 22, replace=False)] = True
                masks = np.stack(
                    (point, (lag >= 30) & (lag < 42), lag >= 60, lag >= 24)
                )
                for node, site in enumerate(panel.station_ids):
                    clean = panel.values[origin - 71 : origin + 1, node, col]
                    target = panel.values[origin + np.array(HORIZONS), node, col]
                    if not (np.isfinite(clean).all() and np.isfinite(target).all()):
                        continue
                    linear = np.stack(
                        [np.interp(lag, lag[~m], clean[~m]) for m in masks]
                    )
                    surface = model.surface[origin - 71 : origin + 1]
                    base = repair(sample, model, surface, node, variable, masks)[0]
                    local = repair(
                        sample,
                        model,
                        surface,
                        node,
                        variable,
                        masks,
                        local_context=True,
                    )[0]
                    if not np.array_equal(np.isfinite(base), np.isfinite(local)):
                        raise ValueError("Different repair coverage")
                    histories = [
                        np.broadcast_to(clean, base.shape),
                        linear,
                        base,
                        local,
                    ]
                    estimates = []
                    for series in histories:
                        features = np.column_stack(
                            (
                                series,
                                np.tile(
                                    calendar(clock[origin : origin + 1]),
                                    (len(masks), 1),
                                ),
                            )
                        )
                        estimates.append(features @ beta + intercept)
                    for m, label in enumerate(
                        ("point", "middle12", "terminal12", "terminal48")
                    ):
                        for j, horizon in enumerate(HORIZONS):
                            rows.append(
                                dict(
                                    region=region,
                                    year=year,
                                    origin=stamp,
                                    station=site,
                                    variable=variable,
                                    fit_windows=n,
                                    mask=label,
                                    horizon=horizon,
                                    target=target[j],
                                    **{
                                        key: pred[m, j]
                                        for key, pred in zip(
                                            ("clean", "linear", "standard", "local"),
                                            estimates,
                                            strict=True,
                                        )
                                    },
                                )
                            )
            print(region, variable, flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluate_forecasts(
        args.raw, pd.read_csv(HERE / "sources.csv", dtype={"site": str})
    ).to_csv(args.output, index=False, mode="x")
