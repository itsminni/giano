"""Evaluate the pre-2022 Ridge fits and paired repair methods on 2022 origins."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from evaluate import past_window, repair

from giano.fill import Predictor
from giano.model import VARIABLES
from giano.trentino import read_network

METHODS = ("clean", "median", "interpolation", "giano1", "giano2", "giano2_prior_year")
MASKS = ("point", "middle12", "terminal12", "terminal48")
FITS = ("1y", "8y", "all-history", "pooled")
HORIZONS = (1, 6, 12, 24)


def distance(a, b, field):
    diff = a - b
    return np.abs((diff + 180) % 360 - 180 if field == "wind_direction" else diff)


def evaluate(data, cases, output, *, local_context=False, station_reference=False):
    panel, surface = read_network(data, "2021-01-01", "2023-01-01", "2022-01-01")
    fields = xr.Dataset({"sp": (("time", "station"), surface.astype(float) * 100)})
    predictors = [Predictor(fields, panel, trentino=b) for b in (True, False)]
    rows, reconstruction = [], []
    for path in sorted(cases.glob("*.npz")):
        with np.load(path) as source:
            c = dict(source)
        site, field = str(c["site"]), str(c["field"])
        with np.load(cases / "pooled" / f"{field}.npz") as pooled:
            coefficients = [*c["coefficients"], pooled["coefficients"]]
            intercepts = [*c["intercepts"], pooled["intercept"]]
            counts = [*c["training_windows"], int(pooled["count"])]
        node, col = panel.station_ids.index(site), VARIABLES.index(field)
        for i, t in enumerate(c["times"]):
            end = panel.times.searchsorted(t)
            sample = past_window(panel, end)
            clean = sample.values[48:120, node, col]
            original = c["repairs"][i]
            if not np.array_equal(clean, original[0, 0]):
                raise ValueError(f"Input disagrees with original station {site}")
            if not np.array_equal(
                panel.values[end + np.array(HORIZONS), node, col], c["targets"][i]
            ):
                raise ValueError("Forecast targets disagree")
            masks = c["masks"][i]
            extra = [
                repair(
                    sample,
                    pred,
                    surface[end - 71 : end + 1],
                    node,
                    field,
                    masks,
                    verify=i == 0,
                    local_context=local_context,
                    station_reference=station_reference,
                )[0]
                for pred in predictors
            ]
            histories = np.concatenate([original, np.stack(extra, axis=1)], axis=1)
            design = np.concatenate(
                (
                    histories,
                    np.broadcast_to(c["calendar"][i], (*histories.shape[:2], 4)),
                ),
                axis=-1,
            )
            for m, name in enumerate(MASKS):
                gap = histories[m][:, masks[m]]
                ok = np.isfinite(gap).all(axis=0)
                record = dict(
                    variable=field,
                    station=site,
                    origin=t,
                    mask=name,
                    hidden=int(masks[m].sum()),
                    paired=int(ok.sum()),
                )
                if ok.any():
                    for method, e in zip(
                        METHODS,
                        distance(gap[:, ok], gap[0, ok], field).mean(axis=1),
                        strict=True,
                    ):
                        record[method] = e
                reconstruction.append(record)
            for f, fit in enumerate(FITS):
                forecasts = design @ coefficients[f] + intercepts[f]
                errors = distance(forecasts, c["targets"][i], field)
                for m, name in enumerate(MASKS):
                    for h, horizon in enumerate(HORIZONS):
                        rows.append(
                            dict(
                                variable=field,
                                station=site,
                                origin=t,
                                mask=name,
                                horizon=horizon,
                                fit=fit,
                                training_windows=counts[f],
                                **dict(zip(METHODS, errors[m, :, h], strict=True)),
                            )
                        )
        print(path.stem, flush=True)
    scores = pd.DataFrame(rows)
    if scores.empty:
        raise ValueError("No prepared station cases")
    scores.to_csv(output / "forecasts.csv", index=False)
    pd.DataFrame(reconstruction).to_csv(output / "reconstruction.csv", index=False)
    summarize(scores).to_csv(output / "scores.csv", index=False, float_format="%.6f")


def summarize(scores):
    scores = scores.assign(month=pd.to_datetime(scores.origin).dt.month)
    tables = []
    for dims in ([], ["mask"], ["horizon"], ["station"], ["month"]):
        keys = ["variable", "fit", *dims]
        common = scores.dropna(subset=list(METHODS))
        counts = scores.groupby(keys).size().rename("scheduled").to_frame()
        counts["paired"] = common.groupby(keys).size()
        station_keys = list(dict.fromkeys([*keys, "station"]))
        stations = common.groupby(station_keys)[list(METHODS)].mean().reset_index()
        result = counts.join(stations.groupby(keys)[list(METHODS)].mean())
        result["stations"] = stations.groupby(keys).size()
        result["wins_vs_v1"] = (
            stations.assign(win=stations.giano2 < stations.giano1)
            .groupby(keys)
            .win.sum()
        )
        sizes = scores.groupby(keys).training_windows.agg(["min", "max"])
        result = result.join(
            sizes.rename(
                columns={"min": "min_training_windows", "max": "max_training_windows"}
            )
        ).reset_index()
        for dim in ("mask", "horizon", "station", "month"):
            if dim not in dims:
                result[dim] = "all"
        tables.append(result)
    return pd.concat(tables, ignore_index=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "cases", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--local-context", action="store_true")
    p.add_argument("--station-reference", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    evaluate(
        args.data,
        args.cases,
        args.output,
        local_context=args.local_context,
        station_reference=args.station_reference,
    )
