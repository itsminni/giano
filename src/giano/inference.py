"""Label-free inference windows using the retained global-outage input recipe."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import INPUTS, VARIABLES
from .physical import physical_prediction
from .windows import (
    HistoryReference,
    LongPanel,
    Outage,
    fit_history,
    nearest_nodes,
    prepare_outage_span,
    window_from_span,
)


def _masked_panel(panel: LongPanel, outage: Outage) -> LongPanel:
    values = panel.values.copy()
    if outage.regime == "other-stations":
        values[outage.start : outage.end, outage.node] = np.nan
    else:
        values[outage.start : outage.end] = np.nan
    auxiliary = panel.auxiliary
    if outage.regime == "no-contemporaneous-inputs":
        auxiliary = auxiliary.copy()
        auxiliary[outage.start : outage.end] = np.nan
    return replace(panel, values=values, auxiliary=auxiliary)


def _cancelled(values: np.ndarray) -> bool:
    angles = np.deg2rad(values[np.isfinite(values)].astype(np.float64))
    return bool(
        len(angles)
        and np.hypot(np.sin(angles).sum(), np.cos(angles).sum()) <= len(angles) * 1e-6
    )


def _ambiguous_direction(
    panel: LongPanel,
    outage: Outage,
    sources: np.ndarray,
    *,
    history_panel: LongPanel | None = None,
) -> np.ndarray:
    """Mark cancellation in untrained wind fallbacks without changing model inputs."""
    ambiguous = np.zeros(outage.end - outage.start, dtype=bool)
    ordered = nearest_nodes(panel, outage.node, len(panel.station_ids))[1:]
    structural = np.asarray(panel.availability.any(axis=0), dtype=bool)
    donors = ordered[structural[ordered, 5]][:8]
    angles = panel.values[outage.start : outage.end, donors, 5].astype(np.float64)
    allowed = panel.availability[outage.start : outage.end, donors, 5]
    angles = np.where(allowed, angles, np.nan)
    radians = np.deg2rad(angles)
    counts = np.isfinite(angles).sum(axis=1)
    cancelled = (
        np.hypot(np.nansum(np.sin(radians), axis=1), np.nansum(np.cos(radians), axis=1))
        <= counts * 1e-6
    )
    ambiguous |= (sources == 1) & (counts >= 3) & cancelled
    if not (sources == 3).any():
        return ambiguous
    past = panel if history_panel is None else history_panel
    history = np.where(
        past.availability[: past.train_end, :, 5],
        past.values[: past.train_end, :, 5],
        np.nan,
    )
    months = panel.times.astype("datetime64[M]").astype(np.int64) % 12
    history_months = (
        past.times[: past.train_end].astype("datetime64[M]").astype(np.int64) % 12
    )
    own = history[:, outage.node]
    finite = own[np.isfinite(own)]
    episode_months = months[outage.start : outage.end]
    for month in np.unique(episode_months[sources == 3]):
        selected = own[history_months == month]
        selected = selected[np.isfinite(selected)]
        if len(selected) < 24:
            selected = finite
            if len(finite) < 4:
                selected = history[history_months == month]
                selected = selected[np.isfinite(selected)]
                if len(selected) < 24:
                    selected = history[np.isfinite(history)]
        ambiguous |= (sources == 3) & (episode_months == month) & _cancelled(selected)
    return ambiguous


class NaturalLongWindows(Dataset[dict[str, torch.Tensor]]):
    """One explicitly masked station episode, disjoint 72-hour output cores."""

    def __init__(
        self,
        panel: LongPanel,
        outage: Outage,
        *,
        max_nodes: int = 24,
        _evaluation_history: Callable[[], HistoryReference] | None = None,
    ):
        outage.validate(panel)
        if type(max_nodes) is not int or not 1 <= max_nodes <= 24:
            raise ValueError("Giano inference requires 1 to 24 station slots")
        self.panel, self.outage = panel, outage
        self.starts: tuple[int, ...] = ()
        self.span: dict[str, torch.Tensor] | None = None
        self.history_sha256: str | None = None
        self.ambiguity = np.zeros(outage.end - outage.start, dtype=bool)
        active = panel.availability[outage.start : outage.end, outage.node]
        missing = ~np.isfinite(panel.values[outage.start : outage.end, outage.node])
        if not bool((active & missing).any()):
            self.uncovered_reason = "no-natural-missing-values"
            return
        lower, upper = 0, len(panel.times)
        left = outage.start - 48
        right = outage.start + ((outage.end - outage.start + 71) // 72) * 72 + 48
        self.uncovered_reason = "insufficient-window-context"
        if left < lower or right > upper:
            return
        # Sanitize before even constructing the old sample's zero-valued label fields.
        masked = _masked_panel(panel, outage)
        # A training outage must always exclude its own hidden observations.
        if _evaluation_history is None or outage.start < panel.train_end:
            reference = fit_history(masked, outage)
        else:
            reference = _evaluation_history()
        span = prepare_outage_span(masked, outage, reference, max_nodes=max_nodes)
        if bool(span["target_mask"].any()):
            raise ValueError("Natural gap windows must not construct target labels")
        offset = outage.start - int(span["span_start"])
        sources = span["baseline_source"][
            offset : offset + outage.end - outage.start, 0, 5
        ]
        self.ambiguity = _ambiguous_direction(masked, outage, sources.numpy())
        self.span, self.history_sha256 = span, reference.sha256
        self.starts = tuple(range(outage.start, outage.end, 72))
        self.uncovered_reason = "none"

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.span is None:
            raise IndexError(index)
        sample = window_from_span(self.span, self.starts[index], 72)
        return {k: sample[k] for k in (*INPUTS, "core_start")}


@dataclass(frozen=True)
class LongReconstruction:
    values: np.ndarray
    count: np.ndarray
    corrected: np.ndarray
    baseline_source: np.ndarray
    circular_ambiguity: np.ndarray
    forward_batches: int


@torch.inference_mode()
def reconstruct_long(
    model: Any, windows: NaturalLongWindows, *, batch_size: int = 8
) -> LongReconstruction:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Giano inference batch size must be positive")
    outage = windows.outage
    shape = (outage.end - outage.start, 6)
    values = np.full(shape, np.nan, dtype=np.float64)
    count = np.zeros(shape, dtype=np.uint8)
    corrected = np.zeros(shape, dtype=bool)
    sources = np.zeros(shape, dtype=np.uint8)
    active = windows.panel.availability[outage.start : outage.end, outage.node]
    missing = ~np.isfinite(windows.panel.values[outage.start : outage.end, outage.node])
    batches = 0
    model.eval()
    for batch in DataLoader(
        windows, batch_size=batch_size, shuffle=False, num_workers=0
    ):
        prediction, residual = model(**{k: batch[k] for k in INPUTS})
        if (
            prediction.shape != batch["values"].shape
            or residual.shape != prediction.shape
            or not bool(
                torch.isfinite(prediction).all() & torch.isfinite(residual).all()
            )
        ):
            raise ValueError("Invalid gap-filling inference output")
        for j, variable in enumerate(VARIABLES):
            decoded = physical_prediction(
                prediction[:, :, :1, j],
                batch["center"][:, :1, j],
                batch["scale"][:, :1, j],
                variable=variable,
            )[..., 0].numpy()
            for row, start in enumerate(batch["core_start"].tolist()):
                offset = start - outage.start
                length = min(72, shape[0] - offset)
                positions = np.arange(offset, offset + length)
                selected = active[positions, j] & missing[positions, j]
                positions = positions[selected]
                if bool(count[positions, j].any()):
                    raise ValueError("Output cores must be disjoint")
                values[positions, j] = decoded[row, :length][selected]
                corrected[positions, j] = (
                    residual[row, :length, 0, j].abs().numpy()[selected] > 1e-7
                )
                sources[positions, j] = batch["baseline_source"][
                    row, :length, 0, j
                ].numpy()[selected]
                count[positions, j] = 1
        batches += 1
    ambiguous = windows.ambiguity & ~corrected[:, 5] & (count[:, 5] > 0)
    values[ambiguous, 5] = np.nan
    count[ambiguous, 5] = 0
    return LongReconstruction(values, count, corrected, sources, ambiguous, batches)
