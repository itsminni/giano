"""Prepare station-level Ridge learning curves in the Giano v1.0.0 environment."""

import argparse
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import torch
from learning_curve_fit import HORIZONS, calendar, pooled_fit

ORIGINAL = {
    "temperature": "T0088",
    "humidity": "T0118",
    "pressure": "T0103",
    "precipitation": "T0094",
    "wind_direction": "T0442",
    "wind_speed": None,
}


def prepare(data, checkpoints, output, field):
    dataset = import_module("giano.spatiotemporal_dataset")
    regression = import_module("giano.downstream.ridge_forecaster")
    repairer = import_module("giano.evaluation.forecasting.checkpoint_repairer")
    protocol = import_module("giano.evaluation.forecasting.protocol")
    panel = dataset.build_spatiotemporal_panel(data / "2-processed-v2", field)
    clock = panel.times
    end = clock.searchsorted(np.datetime64("2022-01-01"))
    stop = clock.searchsorted(np.datetime64("2023-01-01"))
    starts = [
        clock.searchsorted(np.datetime64(d)) for d in ("2021-01-01", "2014-01-01")
    ]
    parts = []
    for site in sorted(panel.station_ids, key=lambda s: (s != ORIGINAL[field], s)):
        values = panel.values[:, panel.station_ids.index(site)].astype(float)
        origins = regression.complete_origins(
            values, history_length=72, horizons=HORIZONS, start=0, end=end
        )
        if (origins >= starts[0] + 71).sum() < 500:
            continue
        # Four disjoint weekly slots per month, rotating UTC/source hours to sample the daily cycle.
        valid = regression.complete_origins(
            values, history_length=72, horizons=HORIZONS, start=end + 71, end=stop
        )
        days = clock[valid].astype("datetime64[D]")
        months = days.astype("datetime64[M]")
        day_of_month = (days - months).astype(int)
        weeks = day_of_month // 7
        eligible = (weeks < 4) & (
            clock[valid].astype("datetime64[h]").astype(int) % 24 == weeks * 6
        )
        slots = months.astype(int) * 4 + weeks
        _, idx = np.unique(slots[eligible], return_index=True)
        test = valid[eligible][idx]
        if len(np.unique(clock[test].astype("datetime64[M]"))) < 8:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(values, 72)
        betas, biases, counts = [], [], []
        for first in (*starts, None):
            ix = origins if first is None else origins[origins >= first + 71]
            design = np.column_stack((windows[ix - 71], calendar(clock[ix])))
            targets = values[ix[:, None] + HORIZONS]
            ridge = regression.RidgeRegressor(1.0).fit(design, targets)
            beta = ridge.coefficients / ridge.x_scale[:, None]
            betas.append(beta)
            biases.append(ridge.y_mean - ridge.x_mean @ beta)
            counts.append(len(ix))
            if first is None:
                full = np.column_stack((design, targets))
                center = full.mean(axis=0)
                full -= center
                parts.append((len(full), center, full.T @ full))
                del full
            del design
        legacy = repairer.checkpoint_history_repairer(
            checkpoints
            / "corrected_v2_spectral"
            / "imputeformer"
            / "seed-42"
            / f"{field}.pt",
            panel,
            site,
            family="imputeformer",
            device=torch.device("cpu"),
        )
        masks, repairs = [], []
        for t in test:
            rng = np.random.default_rng(
                42 + int(clock[t].astype("datetime64[h]").astype(int))
            )
            point = np.zeros(72, bool)
            point[rng.choice(72, 22, replace=False)] = True
            pos = np.arange(72)
            m = np.stack((point, (pos >= 30) & (pos < 42), pos >= 60, pos >= 24))
            clean = windows[t - 71]
            row = []
            for hidden in m:
                median = np.where(hidden, np.median(clean[~hidden]), clean)
                interp = (
                    protocol._circular_interpolation
                    if field == "wind_direction"
                    else protocol._linear_interpolation
                )
                row.append(
                    [
                        clean,
                        median,
                        interp(clean, hidden),
                        legacy(median, ~hidden, clock[t - 71 : t + 1]),
                    ]
                )
            repairs.append(row)
            masks.append(m)
        np.savez_compressed(
            output / f"{field}_{site}.npz",
            field=field,
            site=site,
            times=clock[test],
            calendar=calendar(clock[test]),
            targets=values[test[:, None] + HORIZONS],
            masks=masks,
            repairs=repairs,
            coefficients=betas,
            intercepts=biases,
            training_windows=counts,
        )
        print(field, site, counts, len(test), flush=True)
        if len(parts) == 4:
            break
    if parts:
        beta, intercept, n = pooled_fit(parts)
        pooled = output / "pooled"
        pooled.mkdir(exist_ok=True)
        np.savez(
            pooled / f"{field}.npz", coefficients=beta, intercept=intercept, count=n
        )
    else:
        print(field, "no eligible stations", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "checkpoints", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if find_spec("giano.spatiotemporal_dataset") is None:
        p.error(
            "Run this exporter in the Giano v1.0.0 environment (see https://github.com/itsminni/giano/wiki/Downstream-Utility)"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    for variable in ORIGINAL:
        prepare(args.data, args.checkpoints, args.output, variable)
