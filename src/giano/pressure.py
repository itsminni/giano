"""Auxiliary surface pressure calibrated to station pressure."""

from dataclasses import dataclass

import numpy as np
import torch

from .inference import NaturalLongWindows
from .windows import REGIMES, Outage


@dataclass(frozen=True)
class PressureOffset:
    offset: np.ndarray
    count: np.ndarray
    valid: np.ndarray


def pressure_signal(
    surface: np.ndarray,
    available: np.ndarray,
    rows: np.ndarray,
    nodes: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    reference: PressureOffset,
    outage: Outage,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize mapped surface pressure and honor the no-contemporaneous mask."""
    if (
        surface.ndim != 2
        or available.shape != surface.shape
        or available.dtype != bool
        or rows.ndim != 1
        or nodes.ndim != 1
        or rows.dtype.kind not in "iu"
        or nodes.dtype.kind not in "iu"
        or not len(rows)
        or not len(nodes)
        or np.any(rows >= len(surface))
        or np.any(rows < 0)
        or np.any(nodes < 0)
        or np.any(nodes >= surface.shape[1])
        or len(set(nodes.tolist())) != len(nodes)
        or not (np.diff(rows) == 1).all()
        or center.shape != nodes.shape
        or scale.shape != nodes.shape
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
        or reference.offset.shape != (surface.shape[1],)
        or reference.count.shape != reference.offset.shape
        or reference.valid.shape != reference.offset.shape
        or reference.valid.dtype != bool
        or outage.regime not in REGIMES
    ):
        raise ValueError("Invalid pressure input geometry or calibration")
    source = surface[np.ix_(rows, nodes)]
    mask = np.isfinite(source) & available[np.ix_(rows, nodes)] & reference.valid[nodes]
    if outage.regime == "no-contemporaneous-inputs":
        mask[(rows >= outage.start) & (rows < outage.end)] = False
    physical = (source.astype(np.float64) + reference.offset[nodes]).astype(np.float32)
    values = np.where(
        mask, (physical - center.astype(np.float32)) / scale.astype(np.float32), 0.0
    ).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite normalized pressure signal")
    return values, mask


class PressureWindows(NaturalLongWindows):
    """Supply the training-calibrated pressure covariate on allowed cells."""

    surface: np.ndarray
    offset: PressureOffset

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        if self.span is None:
            raise IndexError(index)
        nodes = self.span["node_indices"]
        valid = nodes >= 0
        start = int(sample["core_start"])
        values, mask = pressure_signal(
            self.surface,
            self.panel.availability[..., 2],
            np.arange(start, start + 72),
            nodes[valid].numpy(),
            sample["center"][valid, 2].numpy(),
            sample["scale"][valid, 2].numpy(),
            self.offset,
            self.outage,
        )
        for k, z in (("auxiliary", values), ("auxiliary_mask", mask)):
            sample[k] = sample[k].clone()
            sample[k][:, valid, 2] = torch.from_numpy(z)
        return sample
