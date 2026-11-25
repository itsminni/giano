"""Score the original release's seed-42 repairs on the 2020 Giano test design.

Run with the v1.0.0 Python environment from the root of the current checkout.
"""

import argparse
import json
from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROTOCOL = Path(__file__).resolve().parents[2] / "src/giano/protocol"
# These are the four variables with paired ground truth in the fixed 2020 study.
FIELDS = ("temperature", "humidity", "pressure", "precipitation")
METHODS = ("linear", "giano1", "bilstm", "giano2")


def year_panel(data, field):
    dataset = import_module("giano.spatiotemporal_dataset")
    full = dataset.build_spatiotemporal_panel(data / "2-processed-v2", field)
    left, right = full.times.searchsorted(
        np.array(["2020-01-01", "2021-01-01"], dtype="datetime64[ns]")
    )
    if right - left != 8784:
        raise ValueError("Expected the full source-hour grid for 2020")
    bounds = full.coverage_bounds.copy() - left
    bounds[:, 0] = bounds[:, 0].clip(0, 8784)
    bounds[:, 1] = bounds[:, 1].clip(-1, 8783)
    return replace(
        full,
        times=full.times[left:right].copy(),
        values=full.values[left:right].copy(),
        auxiliary=None if full.auxiliary is None else full.auxiliary[left:right].copy(),
        coverage_bounds=bounds,
    )


@torch.inference_mode()
def reconstruct(panel, node, start, hours, regime, models, config):
    dataset = import_module("giano.spatiotemporal_dataset")
    prediction = import_module("giano.prediction")
    meteorology = import_module("giano.meteorology")
    sl = slice(start, start + hours)
    values = panel.values.copy()
    if regime == "other-stations":
        values[sl, node] = np.nan
    else:
        values[sl] = np.nan
    auxiliary = panel.auxiliary
    if regime == "no-contemporaneous-inputs" and auxiliary is not None:
        auxiliary = auxiliary.copy()
        auxiliary[sl] = np.nan
    masked = replace(panel, values=values, auxiliary=auxiliary)
    windows = dataset.SpatiotemporalWindowDataset(
        masked,
        "all",
        seq_len=config.seq_len,
        stride=max(1, config.seq_len // 2),
        max_nodes=config.max_nodes,
        min_context_points=config.min_context_points,
        seed=config.seed,
        mask_mode="natural",
        point_rate=(config.point_rate_min, config.point_rate_max),
        block_lengths=config.block_lengths,
    )
    natural = dataset.NaturalGapWindowDataset(windows)
    sums = np.zeros((len(models), hours), np.float32)
    count = np.zeros(hours, int)
    for i, (window, targets) in enumerate(natural.groups):
        t = int(windows.starts[window])
        if node not in targets or t >= sl.stop or t + config.seq_len <= start:
            continue
        sample = natural[i]
        slot = sample["node_indices"].tolist().index(node)
        offsets = np.arange(t - start, t - start + config.seq_len)
        active = (
            sample["evaluation_mask"][:, slot].numpy()
            & (offsets >= 0)
            & (offsets < hours)
        )
        if not active.any():
            continue
        inputs = {
            k: sample[k][None]
            for k in ("features", "coordinates", "node_mask", "baseline")
        }
        for m, model in enumerate(models):
            pred, _ = model(**inputs)
            decoded = prediction.physical_prediction(
                pred,
                sample["center"][None],
                sample["scale"][None],
                variable=panel.variable,
            )
            sums[m, offsets[active]] += decoded[0, :, slot].numpy()[active]
        count[offsets[active]] += 1
    interp = dataset._temporal_baseline(values[:, node : node + 1], panel.variable)[
        sl, 0
    ].clip(*meteorology.PHYSICAL_BOUNDS_BY_VARIABLE[panel.variable])
    native = count > 0
    filled = np.broadcast_to(interp, sums.shape).copy()
    filled[:, native] = sums[:, native] / count[native]
    return interp, filled, native


def evaluate(data, checkpoints, predictions, out):
    imputeformer = import_module("giano.model.train_imputeformer")
    bilstm = import_module("giano.baselines.train_bilstm")
    impute = import_module("giano.impute")
    spec = json.loads((PROTOCOL / "historical.json").read_text())
    rows = []
    with np.load(PROTOCOL / "test-support.npz") as f:
        support = {k: np.unpackbits(f[k], axis=0, count=8784).astype(bool) for k in f}
    for col, field in enumerate(FIELDS):
        panel = year_panel(data, field)
        models = []
        configs = []
        for loader, sub in [
            (
                imputeformer.load_imputeformer_checkpoint,
                "corrected_v2_spectral/imputeformer",
            ),
            (bilstm.load_bilstm_checkpoint, "corrected_v2/bilstm/fair"),
        ]:
            m, meta = loader(
                checkpoints / sub / "seed-42" / f"{field}.pt", torch.device("cpu")
            )
            if meta["variable"] != field:
                raise ValueError("Wrong checkpoint variable")
            configs.append(impute._checkpoint_training_config(meta))
            models.append(m.eval())
        for k in (
            "seq_len",
            "max_nodes",
            "seed",
            "min_context_points",
            "block_lengths",
            "point_rate_min",
            "point_rate_max",
        ):
            if getattr(configs[0], k) != getattr(configs[1], k):
                raise ValueError(f"Different window setting: {k}")
        for episode, (site, date, hours, regime) in enumerate(
            product(spec["sites"], spec["dates"], spec["hours"], spec["regimes"])
        ):
            start = int(panel.times.searchsorted(np.datetime64("2020-" + date)))
            sl = slice(start, start + hours)
            mask = support[site][sl, col]
            if not mask.any():
                continue
            node = panel.station_ids.index(site)
            truth = panel.values[sl, node].astype(float)
            interp, filled, native = reconstruct(
                panel, node, start, hours, regime, models, configs[0]
            )
            current = predictions[f"{site}_{date}_{hours}_{regime}"][:, col]
            paired = (
                mask
                & np.isfinite(current)
                & np.isfinite(truth)
                & np.isfinite(interp)
                & np.isfinite(filled).all(axis=0)
            )
            for portion, keep in [("all", paired), ("native", paired & native)]:
                record = dict(
                    episode=episode,
                    site=site,
                    date=date,
                    hours=hours,
                    regime=regime,
                    variable=field,
                    portion=portion,
                    measured=int((mask & np.isfinite(truth)).sum()),
                    paired=int(keep.sum()),
                )
                for label, y in zip(
                    METHODS,
                    (interp, *filled, current),
                    strict=True,
                ):
                    record[label + "_absolute_error"] = np.abs(
                        y[keep] - truth[keep]
                    ).sum()
                rows.append(record)
        print(field, "complete", flush=True)
    details = pd.DataFrame(rows)
    details.to_csv(out / "episodes.csv", index=False)
    errors = [f"{method}_absolute_error" for method in METHODS]
    tables = []
    for dims in ([], ["hours"], ["regime"], ["hours", "regime"]):
        grouped = (
            details.groupby(["variable", "portion", *dims])[
                ["paired", "measured", *errors]
            ]
            .sum()
            .reset_index()
        )
        for method in METHODS:
            grouped[f"{method}_mae"] = grouped.pop(
                f"{method}_absolute_error"
            ) / grouped.paired.replace(0, np.nan)
        for dim in ("hours", "regime"):
            if dim not in dims:
                grouped[dim] = "all"
        tables.append(grouped)
    pd.concat(tables, ignore_index=True).to_csv(
        out / "scores.csv", index=False, float_format="%.8f"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "checkpoints", "predictions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if find_spec("giano.spatiotemporal_dataset") is None:
        parser.error(
            "Run this comparison in the Giano v1.0.0 environment (see https://github.com/itsminni/giano/wiki/Model-and-evidence)"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    with np.load(args.predictions) as predictions:
        evaluate(args.data, args.checkpoints, predictions, args.output)
