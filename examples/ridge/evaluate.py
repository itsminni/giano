"""Score the packaged Giano ensemble through the frozen rolling-origin Ridge fits."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from giano.fill import Predictor
from giano.inference import _ambiguous_direction
from giano.local_context import apply_local_context
from giano.model import INPUTS, VARIABLES
from giano.physical import physical_prediction
from giano.pressure import pressure_signal
from giano.station_reference import apply_station_reference
from giano.trentino import read_network
from giano.windows import Outage, prepare_outage_span, window_from_span

HOUR = np.timedelta64(1, "h")
VARYING_INPUTS = ("values", "input_mask", "baseline", "baseline_source", "donor_count")


def past_window(panel, end):
    """Copy exactly 72 known time slots. The span builder's margins are empty."""
    history = slice(end - 71, end + 1)

    def padded(a, value):
        return np.pad(a[history], ((48, 48), (0, 0), (0, 0)), constant_values=value)

    return replace(
        panel,
        times=panel.times[end] + np.arange(-119, 49) * HOUR,
        values=padded(panel.values, np.nan),
        auxiliary=padded(panel.auxiliary, np.nan),
        availability=padded(panel.availability, False),
        train_end=0,
    )


def model_inputs(panel, gap, predictor, surface, hidden):
    span = prepare_outage_span(panel, gap, predictor.history, hidden_hours=hidden)
    sample = window_from_span(span, 48, 72)
    nodes = span["node_indices"].numpy()
    real = nodes >= 0
    p, valid = pressure_signal(
        surface,
        panel.availability[48:120, :, 2],
        np.arange(72),
        nodes[real],
        sample["center"].numpy()[real, 2],
        sample["scale"].numpy()[real, 2],
        predictor.offset,
        gap,
    )
    for key, data in (("auxiliary", p), ("auxiliary_mask", valid)):
        sample[key][:, torch.from_numpy(real), 2] = torch.from_numpy(data)
    return {key: sample[key] for key in INPUTS}


def masked_batch(visible, hidden, masks):
    # Physical references, including other stations' donor estimates, are hourly.
    # Choose the fully re-computed state for each hour before temporal attention.
    mask = torch.from_numpy(masks)[:, :, None, None]
    return {
        key: torch.where(mask, hidden[key], value)
        if key in VARYING_INPUTS
        else value.expand(len(masks), *value.shape)
        for key, value in visible.items()
    }


@torch.inference_mode()
def repair(
    panel,
    predictor,
    surface,
    node,
    variable,
    masks,
    verify=False,
    *,
    local_context=False,
    station_reference=False,
):
    gap = Outage(48, 120, node, "other-stations")
    visible = model_inputs(panel, gap, predictor, surface, np.zeros(72, bool))
    hidden = model_inputs(panel, gap, predictor, surface, np.ones(72, bool))
    unique, indices = np.unique(masks, axis=0, return_inverse=True)
    if verify:
        direct = model_inputs(panel, gap, predictor, surface, masks[0])
        batched = masked_batch(visible, hidden, masks[:1])
        if any(not torch.equal(batched[key][0], direct[key]) for key in INPUTS):
            raise RuntimeError(
                "Batch construction differs from directly masking raw observations"
            )
    col = VARIABLES.index(variable)
    ambiguous = (
        _ambiguous_direction(
            panel,
            gap,
            hidden["baseline_source"][:, 0, col].numpy(),
            history_panel=None if predictor.trentino else predictor.panel,
        )
        if variable == "wind_direction"
        else np.zeros(72, bool)
    )
    result = []
    for start in range(0, len(unique), 8):
        mask = unique[start : start + 8]
        batch = masked_batch(visible, hidden, mask)
        predictions = []
        for model in predictor.models:
            normalized, _ = model(**batch)
            decoded = physical_prediction(
                normalized[:, :, :1, col],
                batch["center"][:, :1, col],
                batch["scale"][:, :1, col],
                variable=variable,
            )
            predictions.append(decoded[..., 0].numpy().astype(float))
        mean = np.mean(predictions, axis=0)
        mean[:, ambiguous] = np.nan
        if not predictor.trentino and predictor.history.counts[node, col] < 4:
            mean[:, hidden["baseline_source"][:, 0, col].numpy() == 3] = np.nan
        if local_context:
            for row, missing in enumerate(mask):
                mean[row], _ = apply_local_context(
                    panel, gap, variable, mean[row], hidden=missing
                )
        if station_reference and variable in ("pressure", "wind_direction"):
            for row, missing in enumerate(mask):
                mean[row], _ = apply_station_reference(
                    panel, gap, variable, mean[row], hidden=missing
                )
        measured = panel.values[48:120, node, col]
        result.append(np.where(mask, mean, measured))
    return np.concatenate(result)[indices], hidden


def evaluate(
    data,
    cases,
    variable,
    output,
    reference="packaged",
    *,
    local_context=False,
    station_reference=False,
):
    with np.load(cases / f"{variable}.npz") as inputs:
        cohort = dict(inputs)
    times = cohort["times"]
    # The packaged neural weights and station references predate these origins.
    if times.min() < np.datetime64("2012-01-01") + 72 * HOUR:
        raise ValueError("This protocol requires post-development forecast histories")
    first_history = times.min() - 71 * HOUR
    start = (
        first_history - 365 * 24 * HOUR if reference == "prior-year" else first_history
    )
    panel, surface = read_network(
        data, str(start), str(times.max() + HOUR), str(first_history)
    )
    # read_network returns hPa; the public Predictor expects the external field in Pa.
    fields = xr.Dataset({"sp": (("time", "station"), surface.astype(float) * 100)})
    predictor = Predictor(fields, panel, trentino=reference == "packaged")
    node = panel.station_ids.index(str(cohort["station"]))
    col = VARIABLES.index(variable)
    rows = []
    pressure_totals = []
    for i, stamp in enumerate(times):
        end = int(panel.times.searchsorted(stamp))
        sample = past_window(panel, end)
        if not np.array_equal(sample.values[48:120, node, col], cohort["history"][i]):
            raise RuntimeError("Clean history differs from the original Ridge dataset")
        masks = cohort["masks"][i]
        filled, hidden = repair(
            sample,
            predictor,
            surface[end - 71 : end + 1],
            node,
            variable,
            masks,
            verify=i == 0,
            local_context=local_context,
            station_reference=station_reference,
        )
        if variable == "pressure":
            base = (
                physical_prediction(
                    hidden["baseline"][:, :1, col].unsqueeze(0),
                    hidden["center"][:1, col].reshape(1, 1),
                    hidden["scale"][:1, col].reshape(1, 1),
                    variable=variable,
                )
                .numpy()
                .reshape(72)
            )
            measured = cohort["history"][i].astype(float)
            source = hidden["baseline_source"][:, 0, col].numpy()
            for code in (1, 2, 3):
                eligible = masks & (source == code)
                counts = eligible.sum(axis=0)
                if counts.any():
                    pressure_totals.append(
                        (
                            code,
                            int(counts.sum()),
                            counts @ measured,
                            counts @ base.astype(float),
                            filled[eligible].sum(),
                            np.abs(filled - measured)[eligible].sum(),
                        )
                    )
        design = np.column_stack(
            (filled, np.broadcast_to(cohort["calendar"][i], (len(filled), 4)))
        )
        forecast = ((design - cohort["x_mean"]) / cohort["x_scale"]) @ cohort[
            "coefficients"
        ] + cohort["y_mean"]
        for case_index, mask in enumerate(masks):
            for h, horizon in enumerate((1, 6, 12, 24)):
                rows.append(
                    (
                        str(cohort["station"]),
                        stamp,
                        int(cohort["indices"][i]),
                        42 + case_index // 4,
                        ("point", "block", "terminal", "empirical")[case_index % 4],
                        horizon,
                        int(mask.sum()),
                        int(np.isfinite(filled[case_index, mask]).sum()),
                        float(cohort["targets"][i, h]),
                        *cohort["controls"][i, case_index, :, h],
                        forecast[case_index, h],
                    )
                )
    table = pd.DataFrame(
        rows,
        columns=[
            "station",
            "origin_time",
            "origin_index",
            "seed",
            "mask",
            "horizon",
            "hidden",
            "filled",
            "target",
            "clean",
            "median",
            "interpolation",
            "giano2",
        ],
    )
    if pressure_totals:
        totals = (
            pd.DataFrame(
                pressure_totals,
                columns=[
                    "reference",
                    "hidden_cells",
                    "observed_sum",
                    "reference_sum",
                    "giano2_sum",
                    "absolute_error_sum",
                ],
            )
            .groupby("reference")
            .sum()
        )
        totals.to_csv(output / "pressure-reference.csv")
    table.to_csv(output / f"{variable}.csv", index=False)
    print(
        f"{variable}: {len(table)} forecasts, {table.giano2.count()} finite", flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        choices=("packaged", "prior-year"),
        default="packaged",
        help="Use the packaged calibration or fit local references on the 365 days before the first evaluated history.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=tuple(v for v in VARIABLES if v != "wind_speed"),
        default=[
            "temperature",
            "humidity",
            "pressure",
            "precipitation",
            "wind_direction",
        ],
    )
    parser.add_argument("--local-context", action="store_true")
    parser.add_argument("--station-reference", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    for variable in args.variables:
        evaluate(
            args.data,
            args.cases,
            variable,
            args.output,
            args.reference,
            local_context=args.local_context,
            station_reference=args.station_reference,
        )
