"""Historical-style and fair BiLSTM gap-filling baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn

from giano.spatiotemporal_dataset import INPUT_FEATURES

BiLSTMVariant = Literal["legacy_retrained", "fair"]


@dataclass(frozen=True)
class BiLSTMConfig:
    """Architecture settings for one explicitly named BiLSTM variant."""

    variant: BiLSTMVariant
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    missing_sentinel: float = -100.0

    def validate(self) -> None:
        """Validate architecture values before constructing PyTorch modules."""
        if self.variant not in {"legacy_retrained", "fair"}:
            raise ValueError(f"Unsupported BiLSTM variant: {self.variant}")
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("BiLSTM dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("BiLSTM dropout must be inside [0, 1)")

    @property
    def input_size(self) -> int:
        """Return the feature count for the declared experimental recipe."""
        if self.variant == "legacy_retrained":
            return 2
        return len(INPUT_FEATURES) + 2


class MeteorologicalBiLSTM(nn.Module):
    """Apply a station-wise BiLSTM to common spatiotemporal batches.

    ``legacy_retrained`` intentionally reproduces the historical two-channel
    interface and the ``-100`` station sentinel. ``fair`` consumes every common
    feature plus coordinates, including the explicit observation mask, and
    never places a sentinel in the network.
    """

    def __init__(self, config: BiLSTMConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        recurrent_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=True,
        )
        self.linear = nn.Linear(config.hidden_size * 2, 1)

    def _inputs(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("features must have shape [batch, time, nodes, channels]")
        if features.shape[-1] != len(INPUT_FEATURES):
            raise ValueError(
                f"Expected {len(INPUT_FEATURES)} common features, got "
                f"{features.shape[-1]}"
            )
        if coordinates.shape != (features.shape[0], features.shape[2], 2):
            raise ValueError("coordinates must have shape [batch, nodes, 2]")

        if self.config.variant == "legacy_retrained":
            station = torch.where(
                features[..., 3] > 0.5,
                features[..., 0],
                torch.full_like(features[..., 0], self.config.missing_sentinel),
            )
            auxiliary = torch.where(
                features[..., 4] > 0.5,
                features[..., 2],
                torch.zeros_like(features[..., 2]),
            )
            model_inputs = torch.stack((station, auxiliary), dim=-1)
        else:
            expanded_coordinates = coordinates[:, None, :, :].expand(
                -1, features.shape[1], -1, -1
            )
            model_inputs = torch.cat((features, expanded_coordinates), dim=-1)
        return model_inputs.permute(0, 2, 1, 3).reshape(
            features.shape[0] * features.shape[2],
            features.shape[1],
            self.config.input_size,
        )

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        node_mask: torch.Tensor,
        baseline: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict normalized values through the common gap-filler interface."""
        if node_mask.shape != (features.shape[0], features.shape[2]):
            raise ValueError("node_mask must have shape [batch, nodes]")
        if baseline.shape != features.shape[:3]:
            raise ValueError("baseline must have shape [batch, time, nodes]")
        recurrent, _state = self.lstm(self._inputs(features, coordinates))
        prediction = self.linear(recurrent).squeeze(-1)
        prediction = prediction.reshape(
            features.shape[0], features.shape[2], features.shape[1]
        ).permute(0, 2, 1)
        prediction = prediction * node_mask[:, None, :].to(prediction.dtype)
        return prediction, prediction - baseline

    def get_config(self) -> dict[str, str | int | float]:
        """Return a checkpoint-safe architecture mapping."""
        return asdict(self.config)
