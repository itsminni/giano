"""Replay monthly masked gaps for the station-reference pressure and wind checks."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from giano.fill import Predictor
from giano.station_reference import station_reference
from giano.trentino import PROTOCOL, read_network
from giano.windows import Outage


def evaluate_year(data: Path, year: int) -> pd.DataFrame:
    panel, surface = read_network(
        data, f"{year - 1}-12-01", f"{year + 1}-02-10", f"{year}-01-01"
    )
    predictor = Predictor(xr.Dataset(), panel, trentino=True)
    pressure_sites = json.loads((PROTOCOL / "historical.json").read_text())["sites"]
    rows = []
    for month in range(1, 13):
        start = int(panel.times.searchsorted(np.datetime64(f"{year}-{month:02}-01")))
        for col, variable, durations in (
            (2, "pressure", (1, 6, 12, 168, 720)),
            (5, "wind_direction", (1, 6, 12)),
        ):
            for node, site in enumerate(panel.station_ids):
                if col == 2 and site not in pressure_sites:
                    continue
                observations = panel.values[:, node, col]
                for hours in durations:
                    truth = observations[start : start + hours].astype(float)
                    if not np.isfinite(truth).any():
                        continue
                    window = slice(start - 168, start + ((hours + 71) // 72) * 72 + 48)
                    predictor.panel = replace(
                        panel,
                        **{
                            k: getattr(panel, k)[window]
                            for k in ("times", "values", "availability", "auxiliary")
                        },
                        train_end=0,
                    )
                    predictor.surface = surface[window]
                    gap = Outage(
                        168,
                        168 + hours,
                        node,
                        "no-contemporaneous-inputs" if col == 2 else "other-stations",
                    )
                    baseline = predictor.reconstruct(gap)[0][:, col]
                    paired = np.isfinite(truth) & np.isfinite(baseline)
                    if not paired.any():
                        continue
                    for edge in ("both",) if col == 2 else ("both", "past"):
                        after = (
                            observations[start + hours] if edge == "both" else np.nan
                        )
                        values, source = station_reference(
                            variable,
                            observations[start - 168 : start],
                            after,
                            baseline,
                            hours,
                        )
                        errors = (
                            np.stack([baseline[paired], values[paired]]) - truth[paired]
                        )
                        if col == 5:
                            errors = (errors + 180) % 360 - 180
                        rows.append(
                            (
                                site,
                                month,
                                variable,
                                hours,
                                edge,
                                int(paired.sum()),
                                np.count_nonzero(source[paired]),
                                *np.abs(errors).sum(axis=1),
                            )
                        )
    return pd.DataFrame(
        rows,
        columns=["site", "month", "var", "h", "edge", "paired", "used", "base", "new"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True, choices=(2011, 2012))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    evaluate_year(args.data, args.year).to_csv(args.output, index=False, mode="x")
