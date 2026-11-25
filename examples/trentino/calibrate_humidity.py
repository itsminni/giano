"""Rebuild the humidity columns of the packaged station references from 2009–2010."""

import argparse
from pathlib import Path

import numpy as np

from giano.model import VARIABLES
from giano.trentino import humidity_reference_values
from giano.windows import LongPanel, fit_history


def calibrate(data: Path, base: Path, output: Path):
    with np.load(base, allow_pickle=False) as archive:
        fields = dict(archive)
    stations = tuple(fields["stations"].tolist())
    clock = np.arange(
        np.datetime64("2009-01-01T00"),
        np.datetime64("2011-01-01T00"),
        np.timedelta64(1, "h"),
    )
    col = VARIABLES.index("humidity")
    values = np.full((len(clock), len(stations), len(VARIABLES)), np.nan, np.float32)
    values[..., col] = humidity_reference_values(data, clock, stations)
    panel = LongPanel(
        clock,
        stations,
        fields["coordinates"],
        values,
        np.full_like(values, np.nan),
        np.isfinite(values),
        len(clock),
    )
    ref = fit_history(panel)
    for key in ("center", "scale", "monthly", "counts"):
        fields[key][..., col] = getattr(ref, key)[..., col]
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            stations=fields["stations"],
            coordinates=fields["coordinates"],
            center=fields["center"],
            scale=fields["scale"],
            monthly=fields["monthly"],
            counts=fields["counts"],
            offset=fields["offset"],
            pressure_valid=fields["pressure_valid"],
            pressure_count=fields["pressure_count"],
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "base", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    calibrate(args.data, args.base, args.output)
