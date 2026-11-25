"""Reproduce the 2011 local-reference selection and the separate 2024 calendar check."""

import argparse
from dataclasses import replace
from itertools import groupby, product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from giano.fill import Predictor
from giano.local_context import FIELDS, local_estimates
from giano.trentino import PROTOCOL, read_network
from giano.windows import REGIMES, Outage, Regime

# Equally spaced IDs in the sorted roster after excluding the 2011 development,
# 2020 historical and 2022 downstream target stations. Fixed before scoring 2024.
CHECK_STATIONS = (
    "T0015",
    "T0063",
    "T0090",
    "T0114",
    "T0149",
    "T0178",
    "T0210",
    "T0322",
    "T0363",
    "T0374",
    "T0392",
    "T0405",
    "T0415",
    "T0425",
    "T0437",
    "T0994",
)


def development_episodes(panel):
    with np.load(PROTOCOL / "val.npz") as archive:
        windows = archive["windows"]
        quality = np.unpackbits(archive["quality"], axis=1).astype(bool)
    shift = int((panel.times[0] - np.datetime64("2009-01-01")) / np.timedelta64(1, "h"))
    for key, members in groupby(range(len(windows)), lambda i: tuple(windows[i, :4])):
        node, start, end, regime = map(int, key)
        length = end - start
        if length != 168 and regime != 0:
            continue
        mask = np.zeros((length, 6), bool)
        for i in members:
            t = windows[i, 4] - start
            mask[t : t + 72] = quality[i, : min(72, length - t)]
        durations = [1, 6] if length == 168 else []
        if regime == 0:
            durations.append(length)
        for hours in durations:
            yield node, start - shift, hours, REGIMES[regime], mask[:hours]


def calendar_episodes(panel):
    durations: list[tuple[Regime, int]] = [
        (regime, h) for regime in REGIMES for h in (1, 6)
    ]
    durations.extend(("other-stations", h) for h in (168, 720, 2160))
    for site, date in product(CHECK_STATIONS, ("2024-02-15", "2024-07-15")):
        start = int(panel.times.searchsorted(np.datetime64(date)))
        for regime, hours in durations:
            yield (
                panel.station_ids.index(site),
                start,
                hours,
                regime,
                np.ones((hours, 6), bool),
            )


def evaluate(data, development):
    bounds = (
        ("2010-12-01", "2012-01-10", "2011-01-01")
        if development
        else ("2024-01-01", "2025-01-01", "2024-01-01")
    )
    panel, surface = read_network(data, *bounds)
    predictor = Predictor(xr.Dataset(), panel, True)
    cases = development_episodes(panel) if development else calendar_episodes(panel)
    rows = []
    for node, start, hours, regime, quality in cases:
        # Keep the full correction lookback and the complete neural output window.
        left = max(0, start - 168)
        right = start + ((hours + 71) // 72) * 72 + 48
        local = replace(
            panel,
            **{
                name: getattr(panel, name)[left:right]
                for name in ("times", "values", "auxiliary", "availability")
            },
            train_end=0,
        )
        predictor.panel = local
        predictor.surface = surface[left:right]
        gap = Outage(start - left, start - left + hours, node, regime)
        selected, methods, _, _, baseline, _ = predictor.reconstruct(
            gap, local_context=not development
        )
        for col, variable in enumerate(FIELDS):
            truth = panel.values[start : start + hours, node, col].astype(float)
            reference = baseline[:, col]
            common = quality[:, col] & np.isfinite(reference) & np.isfinite(truth)
            if development:
                bridge, spatial = local_estimates(local, gap, variable)
                proposal = bridge if hours <= 6 else spatial
                used = np.isfinite(proposal) & np.isfinite(reference)
                estimate = np.where(used, proposal, reference)
            else:
                estimate = selected[:, col]
                used = np.isin(
                    methods[:, col], ["station-interpolation", "offset-neighbors"]
                )
            if not np.array_equal(np.isfinite(estimate), np.isfinite(reference)):
                raise ValueError("Different coverage between the two estimators")
            rows.append(
                dict(
                    site=panel.station_ids[node],
                    start=panel.times[start],
                    regime=regime,
                    hours=hours,
                    variable=variable,
                    paired=common.sum(),
                    used=(used & common).sum(),
                    base_absolute_error=np.abs(reference[common] - truth[common]).sum(),
                    candidate_absolute_error=np.abs(
                        estimate[common] - truth[common]
                    ).sum(),
                )
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--year", type=int, choices=(2011, 2024), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluate(args.data, args.year == 2011).to_csv(args.output, index=False, mode="x")
