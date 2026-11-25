"""Compare forecasts at identical stations, origins, masks and horizons."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["station", "origin_time", "origin_index", "seed", "mask", "horizon"]


def attach_original(table, path):
    records = pd.DataFrame(json.loads(path.read_text())["records"]).rename(
        columns={"mask_type": "mask"}
    )
    records["origin_time"] = pd.to_datetime(records["origin_time"])
    if not records.groupby(KEY)["target"].nunique().eq(1).all():
        raise ValueError("Original methods disagree on the target")
    original = records.pivot(index=KEY, columns="method", values="prediction")
    paired = table.set_index(KEY)
    if not original.index.sort_values().equals(paired.index.sort_values()):
        raise ValueError(f"Different experiments in {path}")
    original = original.reindex(paired.index)
    np.testing.assert_allclose(
        records.groupby(KEY)["target"].first().reindex(paired.index),
        paired.target,
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        original[["clean", "corrupted", "interpolation"]],
        paired[["clean", "median", "interpolation"]],
        rtol=0,
        atol=1e-7,
    )
    return paired.join(
        original[["bilstm_fair", "giano"]].rename(columns={"giano": "giano1"})
    ).reset_index()


def scores(variable, table):
    methods = [
        c
        for c in (
            "clean",
            "median",
            "interpolation",
            "bilstm_fair",
            "giano1",
            "giano2",
            "giano2_prior_year",
            "giano2_local_context",
            "giano2_station_context",
        )
        if c in table
    ]
    rows = []
    for mask in ("all", "point", "block", "terminal", "empirical"):
        subset = table if mask == "all" else table[table["mask"] == mask]
        for horizon in (0, 1, 6, 12, 24):
            group = subset if horizon == 0 else subset[subset.horizon == horizon]
            common = group.dropna(subset=methods)
            errors = common[methods].sub(common.target, axis=0)
            if variable == "wind_direction":
                errors = (errors + 180) % 360 - 180
            rows.append(
                dict(
                    variable=variable,
                    station=table.station.iloc[0],
                    mask=mask,
                    horizon=horizon or "all",
                    forecasts=len(group),
                    paired=len(common),
                    **errors.abs().mean().to_dict(),
                )
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--historical",
        type=Path,
        help="Directory with the original downstream/<variable>.json reports",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--local-context-results",
        type=Path,
        help="Include forecasts generated with --local-context and packaged calibration",
    )
    parser.add_argument(
        "--prior-year",
        type=Path,
        help="Include forecasts generated with --reference prior-year",
    )
    parser.add_argument(
        "--station-context-results",
        type=Path,
        help="Forecasts with both --local-context and --station-reference",
    )
    args = parser.parse_args()
    rows = []
    for name in (
        "temperature",
        "humidity",
        "pressure",
        "precipitation",
        "wind_direction",
    ):
        table = pd.read_csv(args.results / f"{name}.csv", parse_dates=["origin_time"])
        if table.duplicated(KEY).any() or len(table) != 16000:
            raise ValueError(f"Unexpected number of paired forecasts for {name}")
        if args.historical:
            table = attach_original(table, args.historical / f"{name}.json")
        for directory, column in (
            (args.prior_year, "giano2_prior_year"),
            (args.local_context_results, "giano2_local_context"),
            (args.station_context_results, "giano2_station_context"),
        ):
            if directory is None:
                continue
            alternative = pd.read_csv(
                directory / f"{name}.csv", parse_dates=["origin_time"]
            ).set_index(KEY)
            original = table.set_index(KEY)
            controls = ["target", "hidden", "clean", "median", "interpolation"]
            pd.testing.assert_frame_equal(
                original[controls].sort_index(),
                alternative[controls].sort_index(),
                check_exact=True,
            )
            table = original.join(
                alternative.giano2.rename(column), validate="one_to_one"
            ).reset_index()
        rows.extend(scores(name, table))
    pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.8f")
