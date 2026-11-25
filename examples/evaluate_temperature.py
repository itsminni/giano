"""Reproduce the paired temperature comparison on three Swiss NOAA stations."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from giano.data import load_input
from giano.fill import fill

DATA = Path(__file__).resolve().parent / "noaa-2024"


def compare(data: xr.Dataset, ensemble: Path | None = None) -> pd.DataFrame:
    """Return every scheduled target, including unscored hours, for inspection."""
    frames = []
    clock = data.time.values
    positions = np.arange(clock.size)
    for month in (3, 6, 9):
        start = np.datetime64(f"2024-{month:02d}-01T00")
        for hours in (6, 168):
            end = start + np.timedelta64(hours, "h")
            inside = (clock >= start) & (clock < end)
            for node, station in enumerate(data.station.values):
                masked = data.copy(deep=True)
                for field in ("temperature", "wind_speed", "wind_direction"):
                    masked[field].values[inside, node] = np.nan
                output = fill(
                    masked, str(start), str(end), ["temperature"], ensemble=ensemble
                )
                series = masked.temperature.values[:, node]
                visible = np.isfinite(series)
                linear = np.interp(
                    positions[inside],
                    positions[visible],
                    series[visible],
                    left=np.nan,
                    right=np.nan,
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "start": start,
                            "hours": hours,
                            "station": station,
                            "time": clock[inside],
                            "observed": data.temperature.values[inside, node],
                            "giano": output.temperature_filled.values[:, node],
                            "linear": linear,
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True)


def scores(rows: pd.DataFrame) -> pd.DataFrame:
    """Weight each hour with a finite observation and both estimates equally."""
    common = rows.dropna(subset=["observed", "giano", "linear"]).copy()
    for method in ("giano", "linear"):
        common[f"{method}_mae"] = (common[method] - common["observed"]).abs()
    grouped = common.groupby("hours")
    result = grouped.agg(
        cells=("time", "size"),
        giano_mae=("giano_mae", "mean"),
        linear_mae=("linear_mae", "mean"),
    )
    result["episodes"] = (
        common.drop_duplicates(["hours", "start", "station"]).groupby("hours").size()
    )
    return result[["episodes", "cells", "giano_mae", "linear_mae"]]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, help="Use a retrained ensemble.pt")
    args = parser.parse_args()
    torch.set_num_threads(2)
    data = load_input(DATA / "observations.csv", DATA / "stations.csv")
    print(scores(compare(data, args.ensemble)).round(4).to_string())
