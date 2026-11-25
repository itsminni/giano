"""Rebuild the official masked training and development windows from Drive data."""

import argparse
import json
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .model import INPUTS
from .pressure import PressureOffset, pressure_signal
from .trentino import PROTOCOL, read_network
from .windows import (
    REGIMES,
    Outage,
    fit_history,
    prepare_outage_span,
    sample_fingerprint,
    window_from_span,
)


def prepare(data: Path, output: Path, *, preview: bool = False) -> None:
    if output.exists():
        raise FileExistsError(f"Choose a new cache directory: {output}")
    panel, surface = read_network(data, "2009-01-01", "2012-01-01", "2011-01-01")
    with np.load(PROTOCOL / "pressure.npz") as stored:
        offsets = [
            PressureOffset(*r)
            for r in zip(
                stored["offsets"], stored["counts"], stored["valid"], strict=True
            )
        ]

    @lru_cache(maxsize=24)
    def reference(gap):
        return fit_history(panel, gap)

    output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        cache = Path(tmp) / "windows"
        cache.mkdir()
        for split in ("train", "val"):
            folder = cache / split
            folder.mkdir()
            with np.load(PROTOCOL / f"{split}.npz") as protocol:
                windows = protocol["windows"]
                ids = np.arange(min(8, len(windows)) if preview else len(windows))
                masks = np.unpackbits(protocol["quality"], axis=1).astype(bool)
                hashes = protocol["sample_hash"]
                np.save(folder / "legacy.npy", protocol["legacy"][ids])
            arrays = {}
            for key, members in groupby(ids, key=lambda i: tuple(windows[i, :4])):
                node, start, end, regime = map(int, key)
                gap = Outage(start, end, node, REGIMES[regime])
                ref_gap = (
                    None
                    if split == "val"
                    else gap
                    if regime == 0
                    else Outage(start, end, 0, "external-only")
                )
                span = prepare_outage_span(panel, gap, reference(ref_gap))
                nodes = span["node_indices"].numpy()
                for i in members:
                    core, calibration = map(int, windows[i, 4:])
                    sample = window_from_span(span, core, 72)
                    if sample_fingerprint(sample) != hashes[i].tobytes().hex():
                        raise ValueError(
                            f"Source window differs from frozen protocol at {split} row {i}"
                        )
                    p, valid = pressure_signal(
                        surface,
                        panel.availability[..., 2],
                        np.arange(core, core + 72),
                        nodes,
                        sample["center"][:, 2].numpy(),
                        sample["scale"][:, 2].numpy(),
                        offsets[calibration],
                        gap,
                    )
                    sample["auxiliary"][..., 2] = torch.from_numpy(p)
                    sample["auxiliary_mask"][..., 2] = torch.from_numpy(valid)
                    sample["target_mask"] &= torch.from_numpy(masks[i])
                    for field in ("target", "target_physical"):
                        sample[field] = torch.where(
                            sample["target_mask"], sample[field], 0.0
                        )
                    for name in (*INPUTS, "target", "target_mask", "target_physical"):
                        a = sample[name].numpy()
                        if name not in arrays:
                            arrays[name] = np.lib.format.open_memmap(
                                folder / f"{name}.npy",
                                mode="w+",
                                dtype=a.dtype,
                                shape=(len(ids), *a.shape),
                            )
                        arrays[name][i] = a
            for name in arrays:
                arrays[name].flush()
            arrays.clear()
            print(f"{split}: verified {len(ids)} windows", flush=True)
        (cache / "recipe.json").write_text(
            json.dumps({"preview": preview, "protocol": 1})
        )
        cache.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preview", action="store_true", help="Prepare eight real windows per split"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    prepare(args.data, args.output, preview=args.preview)
