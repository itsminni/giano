"""Compare short and extended Ridge inputs on the original forecast cases."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from giano.fill import Predictor
from giano.local_context import apply_local_context
from giano.model import VARIABLES
from giano.station_reference import apply_station_reference
from giano.trentino import read_network
from giano.windows import Outage, nearest_nodes

from .evaluate import HOUR, past_window, repair
from .learning_curve_fit import HORIZONS, calendar

LAGS = np.arange(-71, 1)
MASKS = ("point", "block", "terminal", "empirical")
FIELDS = ("temperature", "humidity", "pressure", "precipitation", "wind_direction")


def channels(values, direction):
    if direction:
        angles = np.deg2rad(values)
        return np.stack((np.sin(angles), np.cos(angles)), axis=-1)
    return values[..., None]


def angle_features(angles):
    pairs = channels(angles, True)
    return pairs.reshape(len(angles), -1)


def context_features(panel, node, variable):
    """Every row describes only hours available by that origin, excluding masked lags."""
    col = VARIABLES.index(variable)
    past = pd.DataFrame(
        channels(panel.values[:, node, col], variable == "wind_direction")
    )
    # Older intervals are disjoint and precede all 72 potentially corrupted readings.
    parts = [
        past.shift(72).rolling(96, min_periods=1).mean(),
        past.shift(168).rolling(552, min_periods=1).mean(),
    ]
    neighbours = nearest_nodes(panel, node, 4)[1:]
    for j, field in enumerate(VARIABLES):
        x = channels(panel.values[:, neighbours, j], field == "wind_direction")
        count = np.isfinite(x).sum(axis=1)
        pooled = np.divide(
            np.nansum(x, axis=1, dtype=float),
            count,
            out=np.full(count.shape, np.nan),
            where=count > 0,
        )
        frame = pd.DataFrame(pooled)
        parts.extend((frame, frame.rolling(24, min_periods=1).mean()))
    return pd.concat(parts, axis=1).to_numpy(dtype=float)


def fit(histories, dates, targets, context):
    design = np.column_stack((histories, dates, context))
    count = np.isfinite(design).sum(axis=0)
    center = np.divide(
        np.nansum(design, axis=0), count, out=np.zeros(design.shape[1]), where=count > 0
    )
    centered = np.nan_to_num(design - center)
    scale = centered.std(axis=0)
    scale[scale < 1e-12] = 1.0
    x = centered / scale
    target_mean = targets.mean(axis=0)
    weights = np.linalg.solve(
        x.T @ x + np.eye(x.shape[1]), x.T @ (targets - target_mean)
    )
    return center, scale, weights, target_mean


def interpolate(history, masks, direction):
    result = np.tile(history.astype(float), (len(masks), 1))
    for row, hidden in zip(result, masks, strict=True):
        visible = np.flatnonzero(~hidden & np.isfinite(row))
        support = row[visible]
        if direction:
            support = np.unwrap(np.deg2rad(support))
        values = np.interp(np.flatnonzero(hidden), visible, support)
        row[hidden] = np.rad2deg(values) % 360 if direction else values
    return result


def repairs_with_prehistory(panel, end, node, variable, masks, point):
    """Give both repairers another seven observed days before the masked history."""
    sl = slice(end - 239, end + 1)
    bounded = replace(
        panel,
        **{
            k: getattr(panel, k)[sl]
            for k in ("times", "values", "availability", "auxiliary")
        },
        train_end=0,
    )
    gap = Outage(0, 240, node, "other-stations")
    series = bounded.values[:, node, VARIABLES.index(variable)]
    removed = np.pad(masks, ((0, 0), (168, 0)))
    linear = interpolate(series, removed, variable == "wind_direction")[:, -72:]
    predictions = np.pad(point, ((0, 0), (168, 0)), constant_values=np.nan)
    for i, missing in enumerate(removed):
        values, _ = apply_local_context(
            bounded, gap, variable, predictions[i], hidden=missing
        )
        predictions[i], _ = apply_station_reference(
            bounded, gap, variable, values, hidden=missing
        )
    return linear, np.where(masks, predictions[:, -72:], series[-72:])


def evaluate(data, cases, variable):
    with np.load(cases / f"{variable}.npz") as source:
        cohort = dict(source)
    cutoff = cohort["times"].min() - 71 * HOUR
    # Read 30 days before the training period as warm-up for the earliest origin.
    first = cutoff - (3 * 365 + 30) * 24 * HOUR
    panel, surface = read_network(
        data, str(first), str(cohort["times"].max() + HOUR), str(cutoff)
    )
    nodes = panel.station_ids
    node, col = nodes.index(str(cohort["station"])), VARIABLES.index(variable)
    features = context_features(panel, node, variable)
    forecaster_dates = calendar(panel.times)
    series = panel.values[:, node, col].astype(float)
    train = np.arange(30 * 24, panel.times.searchsorted(cutoff) - max(HORIZONS))
    hist = series[train[:, None] + LAGS]
    targets = series[train[:, None] + np.array(HORIZONS)]
    keep = np.isfinite(hist).all(axis=1) & np.isfinite(targets).all(axis=1)
    if not keep.any():
        raise ValueError(f"No complete training histories for {nodes[node]}")
    train = train[keep]
    older_channels = 4 if variable == "wind_direction" else 2
    contexts = {
        "72h": features[:, :0],
        "30d-local": features[:, :older_channels],
        "72h-network": features[:, older_channels:],
        "30d-network": features,
    }
    fits = {
        name: fit(hist[keep], forecaster_dates[train], targets[keep], extra[train])
        for name, extra in contexts.items()
    }
    if variable == "wind_direction":
        contexts["72h-circular"] = contexts["72h"]
        fits["72h-circular"] = fit(
            angle_features(hist[keep]),
            forecaster_dates[train],
            angle_features(targets[keep]),
            contexts["72h"][train],
        )
    contexts["historical-72h"] = contexts["72h"]
    fits["historical-72h"] = tuple(
        cohort[key] for key in ("x_mean", "x_scale", "coefficients", "y_mean")
    )
    ds = xr.Dataset({"sp": (("time", "station"), surface.astype(float) * 100)})
    predictor = Predictor(ds, panel, trentino=True)
    rows = []
    for i, stamp in enumerate(cohort["times"]):
        end = int(panel.times.searchsorted(stamp))
        history = series[end + LAGS]
        np.testing.assert_array_equal(history, cohort["history"][i])
        dates = forecaster_dates[end]
        np.testing.assert_allclose(dates, cohort["calendar"][i])
        masks = cohort["masks"][i]
        giano, _ = repair(
            past_window(panel, end),
            predictor,
            surface[end - 71 : end + 1],
            node,
            variable,
            masks,
            local_context=True,
            station_reference=True,
            verify=i == 0,
        )
        repairs = {
            "clean": np.broadcast_to(history, masks.shape),
            "interpolation": interpolate(history, masks, variable == "wind_direction"),
            "giano": giano,
        }
        point, _ = repair(
            past_window(panel, end),
            predictor,
            surface[end - 71 : end + 1],
            node,
            variable,
            masks,
        )
        repairs["interpolation-7d"], repairs["giano-7d"] = repairs_with_prehistory(
            panel,
            end,
            node,
            variable,
            masks,
            point,
        )
        for method, repaired in repairs.items():
            if not np.isfinite(repaired).all():
                raise ValueError("Different forecast coverage between repairs")
            for name, (center, scale, weights, target_mean) in fits.items():
                extra = contexts[name][end]
                shared = np.concatenate((dates, extra))
                lags = angle_features(repaired) if name == "72h-circular" else repaired
                design = np.column_stack(
                    (lags, np.broadcast_to(shared, (len(masks), len(shared))))
                )
                pred = np.nan_to_num((design - center) / scale) @ weights + target_mean
                if name == "72h-circular":
                    sine, cosine = pred[:, ::2], pred[:, 1::2]
                    if np.any(np.hypot(sine, cosine) <= 1e-8):
                        raise ValueError("Undefined direction in paired Ridge forecast")
                    pred = np.rad2deg(np.arctan2(sine, cosine)) % 360
                if name == "historical-72h" and method in ("clean", "interpolation"):
                    control = 0 if method == "clean" else 2
                    np.testing.assert_allclose(
                        pred, cohort["controls"][i, :, control], rtol=0, atol=1e-7
                    )
                error = pred - cohort["targets"][i]
                if variable == "wind_direction":
                    error = (error + 180) % 360 - 180
                for k in range(len(masks)):
                    for h, horizon in enumerate(HORIZONS):
                        rows.append(
                            (
                                variable,
                                nodes[node],
                                stamp,
                                42 + k // 4,
                                MASKS[k % 4],
                                horizon,
                                None if name == "historical-72h" else len(train),
                                name,
                                method,
                                abs(error[k, h]),
                            )
                        )
    return pd.DataFrame(
        rows,
        columns="variable station origin seed mask horizon training_windows design method absolute_error".split(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "cases", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    summaries = []
    for field in FIELDS:
        results = evaluate(args.data, args.cases, field)
        results.to_csv(args.output / f"{field}.csv", index=False)
        summary = results.groupby(
            ["variable", "station", "training_windows", "design", "method"],
            dropna=False,
        ).absolute_error.agg(forecasts="size", mae="mean")
        summaries.append(summary)
        print(summary, flush=True)
    pd.concat(summaries).to_csv(args.output / "context-scores.csv", float_format="%.6f")
