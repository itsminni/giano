"""Score the frozen Giano release on a later DWD outage cohort."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import xarray as xr

from giano.data import make_panel
from giano.fill import Predictor
from giano.model import VARIABLES
from giano.physical import PHYSICAL_BOUNDS_BY_VARIABLE
from giano.windows import Outage

from ..swiss.evaluate import spatial_reference
from .evaluate import PRODUCTS, SCENARIOS, linear_bridge

ROOT = Path(__file__).resolve().parents[2]
ROSTER = Path(__file__).with_name("sources.csv")
DWD = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"
GRID = ("2025-10-01", "2026-11-15")
STARTS = ("2026-01-05T03", "2026-04-05T09", "2026-07-05T15", "2026-09-30T21")
WET = 0.1
INPUTS = {
    "air_temperature": ("TU", PRODUCTS["air_temperature"]),
    "pressure": ("P0", PRODUCTS["pressure"]),
    "precipitation": ("RR", {"R1": "precipitation"}),
    "wind": ("FF", {"F": "wind_speed", "D": "wind_direction"}),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protocol(roster):
    code = sorted((ROOT / "src/giano").glob("*.py"))
    weights = sorted((ROOT / "src/giano/weights").iterdir())
    references = (
        ROSTER,
        Path(__file__),
        ROSTER.with_name("evaluate.py"),
        ROOT / "examples/swiss/evaluate.py",
        ROOT / "uv.lock",
    )
    return {
        "sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (*code, *weights, *references)
            if path.is_file()
        },
        "stations": roster.site.tolist(),
        "grid_utc_end_exclusive": GRID,
        "starts_utc": STARTS,
        "scenarios": [
            (start, regime, hours)
            for start, (regime, hours) in product(STARTS, SCENARIOS)
            if not (start == STARTS[-1] and hours == 2160)
        ],
        "release_options": {"local_context": True, "station_reference": True},
        "methods": ["release", "standard", "linear", "spatial"],
        "precipitation_wet_mm": WET,
        "urls": {
            f"stundenwerte_{prefix}_{site}_akt.zip": f"{DWD}/{category}/recent/stundenwerte_{prefix}_{site}_akt.zip"
            for site, (category, (prefix, _)) in product(roster.site, INPUTS.items())
        },
    }


def read_station(raw, site, prefix, mapping):
    path = raw / f"stundenwerte_{prefix}_{site}_akt.zip"
    if not path.exists():
        return None
    with ZipFile(path) as archive:
        (name,) = [name for name in archive.namelist() if name.startswith("produkt_")]
        with archive.open(name) as stream:
            frame = pd.read_csv(
                stream, sep=";", skipinitialspace=True, na_values=["-999"]
            )
    frame = frame.rename(columns=lambda key: key.strip())
    if not (frame.STATIONS_ID == int(site)).all():
        raise ValueError(f"Unexpected station in {path.name}")
    frame.index = pd.to_datetime(frame.MESS_DATUM.astype(str), format="%Y%m%d%H")
    if not frame.index.is_unique:
        raise ValueError(f"Duplicate hours in {path.name}")
    frame = frame[list(mapping)].rename(columns=mapping)
    for field in mapping.values():
        low, high = PHYSICAL_BOUNDS_BY_VARIABLE[field]
        frame[field] = frame[field].where(frame[field].between(low, high))
    if "wind_direction" in frame:
        frame["wind_direction"] = frame.wind_direction.where(frame.wind_speed > 0)
    return frame


def load_network(raw, roster):
    grid = pd.date_range(*GRID, freq="h", inclusive="left")
    fields = {field: np.full((len(grid), len(roster)), np.nan) for field in VARIABLES}
    for node, station in enumerate(roster.itertuples()):
        for prefix, mapping in INPUTS.values():
            values = read_station(raw, station.site, prefix, mapping)
            if values is not None:
                values = values.reindex(grid)
                for field in mapping.values():
                    fields[field][:, node] = values[field]
    return xr.Dataset(
        {field: (("time", "station"), values) for field, values in fields.items()},
        coords={
            "time": grid,
            "station": roster.site.to_numpy(),
            "latitude": ("station", roster.lat.to_numpy()),
            "longitude": ("station", roster.lon.to_numpy()),
        },
    )


def spatial_bridge(series, node, left, right, direction):
    donors = np.arange(series.shape[1]) != node
    offsets = spatial_reference(series[left - 168 : left], donors, node, direction)
    estimates = series[left:right, donors] + offsets
    count = np.isfinite(estimates).sum(axis=1)
    out = np.full(right - left, np.nan)
    if direction:
        vector = np.nansum(np.exp(1j * np.deg2rad(estimates)), axis=1)
        keep = (count > 0) & (np.abs(vector) > count * 1e-6)
        out[keep] = np.rad2deg(np.angle(vector[keep])) % 360
    else:
        out[count > 0] = np.nanmedian(estimates[count > 0], axis=1)
    return out


def episode(panel, gap, predictions):
    left, right, node = gap.start, gap.end, gap.node
    for col, field in enumerate(VARIABLES):
        series = panel.values[:, :, col].astype(float)
        measured = series[left:right, node]
        direction = field == "wind_direction"
        visible = series[:, node].copy()
        visible[left:right] = np.nan
        if direction:
            keep = np.isfinite(visible)
            visible[keep] = np.rad2deg(np.unwrap(np.deg2rad(visible[keep])))
        estimates = {method: values[:, col] for method, values in predictions.items()}
        estimates["linear"] = linear_bridge(visible, left, right)
        if gap.regime == "other-stations":
            low, high = PHYSICAL_BOUNDS_BY_VARIABLE[field]
            spatial = spatial_bridge(series, node, left, right, direction)
            estimates["spatial"] = spatial.clip(low, high)
        observed = np.isfinite(measured)
        common = observed & np.isfinite(np.stack(list(estimates.values()))).all(axis=0)
        record = {
            "variable": field,
            "measured": int(observed.sum()),
            "paired": int(common.sum()),
        }
        if field == "precipitation":
            record["wet_observed"] = int((measured[common] >= WET).sum())
            record["observed_mm"] = float(measured[common].sum())
        for method, values in estimates.items():
            error = values[common] - measured[common]
            if direction:
                error = (error + 180) % 360 - 180
            record[method + "_covered"] = int((observed & np.isfinite(values)).sum())
            record[method + "_ae"] = float(np.abs(error).sum())
            record[method + "_se"] = float(np.square(error).sum())
            if field == "precipitation":
                rain = values[common]
                record[method + "_wet"] = int((rain >= WET).sum())
                record[method + "_hits"] = int(
                    ((rain >= WET) & (measured[common] >= WET)).sum()
                )
                record[method + "_mm"] = float(rain.sum())
        yield record


def evaluate(raw, roster, lock):
    rows = []
    for region, sites in roster.groupby("region"):
        ds = load_network(raw, sites)
        for stamp in STARTS:
            left = int(ds.time.values.searchsorted(np.datetime64(stamp)))
            panel = make_panel(ds, left)
            model = Predictor(xr.Dataset(), panel, False)
            for start, regime, hours in lock["scenarios"]:
                if stamp != start:
                    continue
                for node in range(len(sites)):
                    gap = Outage(left, left + hours, node, regime)
                    standard = model.reconstruct(gap)[0]
                    release = model.reconstruct(gap, **lock["release_options"])[0]
                    for record in episode(
                        panel, gap, {"release": release, "standard": standard}
                    ):
                        rows.append(
                            dict(
                                region=region,
                                station=panel.station_ids[node],
                                start=stamp,
                                regime=regime,
                                hours=hours,
                                **record,
                            )
                        )
            print(region, stamp, flush=True)
    return pd.DataFrame(rows)


def summarize(episodes):
    groups = episodes.groupby(["regime", "hours", "variable"], sort=True)
    scores = groups.sum(numeric_only=True, min_count=1)
    for method in ("release", "standard", "linear", "spatial"):
        scores[method + "_mae"] = scores.pop(method + "_ae") / scores.paired.replace(
            0, np.nan
        )
        scores[method + "_rmse"] = np.sqrt(
            scores.pop(method + "_se") / scores.paired.replace(0, np.nan)
        )
        scores[method + "_coverage"] = scores.pop(
            method + "_covered"
        ) / scores.measured.replace(0, np.nan)
        wet = scores[method + "_wet"] + scores.wet_observed
        scores[method + "_wet_f1"] = (
            2 * scores.pop(method + "_hits") / wet.replace(0, np.nan)
        )
    return scores.assign(episodes=groups.size())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    roster = pd.read_csv(ROSTER, dtype={"site": str})
    locked = json.loads(json.dumps(protocol(roster)))
    if args.raw is None:
        with args.freeze.open("x") as stream:
            json.dump(locked, stream, indent=2)
        return
    if args.sources is None or args.output is None:
        parser.error("Scoring requires --sources and --output")
    if json.loads(args.freeze.read_text()) != locked:
        raise ValueError("Code, weights or evaluation design changed after freeze")
    sources = pd.read_csv(args.sources, keep_default_na=False)
    if (
        sources.file.duplicated().any()
        or sources.set_index("file").url.to_dict() != locked["urls"]
    ):
        raise ValueError("Source receipt differs from the planned downloads")
    for source in sources.itertuples():
        path = args.raw / source.file
        if source.status == 404:
            if path.exists():
                raise ValueError(f"Unexpected file for unavailable source: {path}")
        elif source.status != 200 or digest(path) != source.sha256:
            raise ValueError(f"Source digest mismatch: {path}")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    result = evaluate(args.raw, roster, locked)
    args.output.mkdir()
    result.to_csv(args.output / "episodes.csv", index=False)
    summarize(result).to_csv(args.output / "scores.csv", float_format="%.8f")


if __name__ == "__main__":
    main()
