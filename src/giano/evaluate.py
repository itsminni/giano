"""Reconstruct and score the 480 fixed 2020 outages on the original Drive data."""

import argparse
import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import torch

from .fill import Predictor
from .inference import reconstruct_long
from .model import VARIABLES
from .trentino import PROTOCOL, read_network, weather_fields
from .windows import Outage


class PhysicalBaseline(torch.nn.Module):
    def forward(self, **inputs):
        baseline = inputs["baseline"]
        return baseline, torch.zeros_like(baseline)


def evaluation_network(data, spec):
    train, _ = read_network(data, "2009-01-01", "2011-01-01", "2011-01-01")
    test, surface = read_network(data, "2020-01-01", "2021-01-01", "2020-01-01")
    panel = replace(
        train,
        **{
            k: np.concatenate([getattr(train, k), getattr(test, k)])
            for k in ("times", "values", "auxiliary", "availability")
        },
    )
    ds = weather_fields(data, panel, surface, spec, start=panel.train_end)
    return ds, panel, surface


def evaluate(
    data: Path,
    ensemble: Path | None = None,
    *,
    predictions: dict[str, np.ndarray] | None = None,
    local_context: bool = False,
    station_reference: bool = False,
):
    spec = json.loads((PROTOCOL / "historical.json").read_text())
    ds, panel, surface = evaluation_network(data, spec)
    model = Predictor(ds, panel, True, ensemble)
    model.surface[panel.train_end :] = surface
    rows, totals = [], []
    baseline = PhysicalBaseline()
    with np.load(PROTOCOL / "test-support.npz") as stored:
        masks = {
            s: np.unpackbits(stored[s], axis=0, count=8784).astype(bool)
            for s in spec["sites"]
        }
    for site, date, hours, regime in product(
        spec["sites"], spec["dates"], spec["hours"], spec["regimes"]
    ):
        node = panel.station_ids.index(site)
        time = np.datetime64(f"2020-{date}")
        start = int(panel.times.searchsorted(time))
        sl = slice(start, start + hours)
        gap = Outage(start, start + hours, node, regime)
        selected, _, _, total, point, windows = model.reconstruct(
            gap, local_context=local_context, station_reference=station_reference
        )
        if predictions is not None:
            predictions[f"{site}_{date}_{hours}_{regime}"] = selected
        physical = reconstruct_long(baseline, windows).values
        truth = panel.values[sl, node].astype(float)
        support = masks[site][start - panel.train_end : start - panel.train_end + hours]
        for j, variable in enumerate(VARIABLES):
            observed = support[:, j] & np.isfinite(truth[:, j])
            paired = (
                observed
                & np.isfinite(selected[:, j])
                & np.isfinite(point[:, j])
                & np.isfinite(physical[:, j])
            )
            row = dict(
                site=site,
                start=str(time),
                hours=hours,
                regime=regime,
                variable=variable,
                measured=int(observed.sum()),
                paired=int(paired.sum()),
            )
            for name, estimated in (
                ("giano", selected),
                ("point", point),
                ("physical", physical),
            ):
                err = estimated[paired, j] - truth[paired, j]
                if variable == "wind_direction":
                    err = (err + 180) % 360 - 180
                row[name + "_absolute_error"] = float(np.abs(err).sum())
            rows.append(row)
        valid_rain = (
            support[:, 3] & np.isfinite(selected[:, 3]) & np.isfinite(truth[:, 3])
        )
        if total is not None and valid_rain.sum() >= 0.95 * hours:
            rain = panel.auxiliary[sl, node, 3][valid_rain].astype(float).clip(0, 300)
            totals.append(
                dict(
                    hours=hours,
                    site=site,
                    start=str(time),
                    measured_mm=truth[valid_rain, 3].sum(),
                    hourly_sum_mm=selected[valid_rain, 3].sum(),
                    independent_mm=rain.sum() * model.calibration["totals"][site],
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Include hourly predictions for paired comparisons in predictions.npz",
    )
    parser.add_argument("--local-context", action="store_true")
    parser.add_argument("--station-reference", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    torch.set_num_threads(2)
    predictions: dict[str, np.ndarray] | None = {} if args.save_predictions else None
    rows, totals = evaluate(
        args.data,
        args.ensemble,
        predictions=predictions,
        local_context=args.local_context,
        station_reference=args.station_reference,
    )
    groups = rows.groupby(["regime", "hours", "variable"])[
        [
            "paired",
            "measured",
            "giano_absolute_error",
            "point_absolute_error",
            "physical_absolute_error",
        ]
    ].sum()
    for method in ("giano", "point", "physical"):
        groups[method + "_mae"] = groups.pop(
            method + "_absolute_error"
        ) / groups.paired.replace(0, np.nan)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        directory = Path(tmp) / "evaluation"
        directory.mkdir()
        rows.to_csv(directory / "episodes.csv", index=False)
        totals.to_csv(directory / "rain-totals.csv", index=False)
        groups.to_csv(directory / "scores.csv")
        if predictions is not None:
            with ZipFile(directory / "predictions.npz", "w", ZIP_DEFLATED) as archive:
                for key, values in predictions.items():
                    with archive.open(f"{key}.npy", "w", force_zip64=True) as entry:
                        np.save(entry, values, allow_pickle=False)
        directory.rename(args.output)
    print(
        f"Scored {len(rows) // len(VARIABLES)} episodes. Read {args.output / 'scores.csv'}"
    )
