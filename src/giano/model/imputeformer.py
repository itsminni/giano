"""Inductive ImputeFormer adaptation for meteorological station networks.

The architecture follows the central ImputeFormer design: projected temporal
attention, efficient embedded spatial attention, alternating temporal/spatial
blocks, and Fourier-domain regularization.  Giano replaces fixed sensor
embeddings with coordinate-conditioned embeddings so the same weights can be
applied to a changing or previously unseen station set.

Reference: Nie et al., "ImputeFormer: Low Rankness-Induced Transformers for
Generalizable Spatiotemporal Imputation", KDD 2024.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ImputeFormerConfig:
    """Serializable architecture parameters."""

    input_dim: int = 12
    input_embedding_dim: int = 32
    spatial_embedding_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    projection_tokens: int = 8
    feed_forward_dim: int = 128
    coordinate_frequencies: int = 6
    dropout: float = 0.1
    max_residual: float = 3.0
    min_learned_gap: int = 5

    @property
    def model_dim(self) -> int:
        """Return the concatenated representation width."""
        return self.input_embedding_dim + self.spatial_embedding_dim

    def validate(self) -> None:
        """Reject invalid or internally inconsistent dimensions."""
        positive = (
            self.input_dim,
            self.input_embedding_dim,
            self.spatial_embedding_dim,
            self.num_heads,
            self.num_layers,
            self.projection_tokens,
            self.feed_forward_dim,
            self.coordinate_frequencies,
            self.min_learned_gap,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("All ImputeFormer dimensions must be positive")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.max_residual <= 0:
            raise ValueError("max_residual must be positive")


class CoordinateEmbedding(nn.Module):
    """Encode latitude/longitude without a fixed station lookup table."""

    frequencies: torch.Tensor

    def __init__(self, output_dim: int, num_frequencies: int) -> None:
        super().__init__()
        frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=True)
        feature_dim = 2 + 4 * num_frequencies
        self.network = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return coordinate features for ``[batch, nodes, 2]`` inputs."""
        phases = coordinates.unsqueeze(-1) * self.frequencies
        encoded = torch.cat(
            [
                coordinates,
                torch.sin(torch.pi * phases).flatten(start_dim=-2),
                torch.cos(torch.pi * phases).flatten(start_dim=-2),
            ],
            dim=-1,
        )
        return self.network(encoded)


class ProjectedTemporalAttention(nn.Module):
    """Compress the temporal axis through learnable latent tokens."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        projection_tokens: int,
        feed_forward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.projector = nn.Parameter(torch.empty(projection_tokens, model_dim))
        nn.init.xavier_uniform_(self.projector)
        self.compress = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.expand = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attention = nn.LayerNorm(model_dim)
        self.norm_feed_forward = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, model_dim),
        )

    def forward(self, values: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Attend over time independently for every station."""
        batch, steps, nodes, width = values.shape
        sequences = values.permute(0, 2, 1, 3).reshape(batch * nodes, steps, width)
        projector = self.projector.unsqueeze(0).expand(batch * nodes, -1, -1)
        latent, _ = self.compress(projector, sequences, sequences, need_weights=False)
        message, _ = self.expand(sequences, latent, latent, need_weights=False)
        sequences = self.norm_attention(sequences + self.dropout(message))
        sequences = self.norm_feed_forward(
            sequences + self.dropout(self.feed_forward(sequences)),
        )
        output = sequences.reshape(batch, nodes, steps, width).permute(0, 2, 1, 3)
        return output * node_mask[:, None, :, None]


class EmbeddedSpatialAttention(nn.Module):
    """Linear-complexity spatial attention driven by station coordinates."""

    def __init__(
        self,
        model_dim: int,
        spatial_embedding_dim: int,
        feed_forward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query = nn.Linear(spatial_embedding_dim, model_dim)
        self.key = nn.Linear(spatial_embedding_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.output = nn.Linear(model_dim, model_dim)
        self.norm_attention = nn.LayerNorm(model_dim)
        self.norm_feed_forward = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, model_dim),
        )

    def forward(
        self,
        values: torch.Tensor,
        spatial_embedding: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Exchange information across stations at every timestamp."""
        valid = node_mask.bool()
        query = torch.softmax(self.query(spatial_embedding), dim=-1)
        key_logits = self.key(spatial_embedding).masked_fill(~valid[..., None], -1e4)
        key = torch.softmax(key_logits, dim=1)
        query = query * valid[..., None]
        projected_values = self.value(values) * valid[:, None, :, None]
        context = torch.einsum("bnd,btne->btde", key, projected_values)
        message = torch.einsum("bnd,btde->btne", query, context)
        message = self.output(message)
        values = self.norm_attention(values + self.dropout(message))
        values = self.norm_feed_forward(
            values + self.dropout(self.feed_forward(values)),
        )
        return values * valid[:, None, :, None]


class MeteorologicalImputeFormer(nn.Module):
    """ImputeFormer adapted to source-optional meteorological observations.

    Missing station values are neutralized before projection and accompanied by
    an explicit mask feature.  The prediction is a residual over temporal
    interpolation, so zero residual exactly reproduces the strong baseline.
    """

    def __init__(self, config: ImputeFormerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ImputeFormerConfig()
        self.config.validate()
        model_dim = self.config.model_dim
        self.input_projection = nn.Linear(
            self.config.input_dim,
            self.config.input_embedding_dim,
        )
        self.coordinate_embedding = CoordinateEmbedding(
            self.config.spatial_embedding_dim,
            self.config.coordinate_frequencies,
        )
        self.temporal_layers = nn.ModuleList()
        self.spatial_layers = nn.ModuleList()
        for _ in range(self.config.num_layers):
            self.temporal_layers.append(
                ProjectedTemporalAttention(
                    model_dim,
                    self.config.num_heads,
                    self.config.projection_tokens,
                    self.config.feed_forward_dim,
                    self.config.dropout,
                ),
            )
            self.spatial_layers.append(
                EmbeddedSpatialAttention(
                    model_dim,
                    self.config.spatial_embedding_dim,
                    self.config.feed_forward_dim,
                    self.config.dropout,
                ),
            )
        self.readout = nn.Sequential(
            nn.Linear(model_dim, self.config.feed_forward_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.feed_forward_dim, 1),
        )
        final_layer = self.readout[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("ImputeFormer readout must end with a linear layer")
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        node_mask: torch.Tensor,
        baseline: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized predictions and residual corrections."""
        if features.ndim != 4:
            raise ValueError("features must have shape [batch, time, nodes, channels]")
        if features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected {self.config.input_dim} features, got {features.shape[-1]}",
            )
        if baseline.shape != features.shape[:-1]:
            raise ValueError("baseline shape must match features without channels")
        missing = (features[..., 3] < 0.5) & node_mask[:, None, :].bool()
        run_lengths = self._missing_run_lengths(missing)
        learned_gap = run_lengths >= self.config.min_learned_gap
        return self.forward_core(
            features,
            coordinates,
            node_mask,
            baseline,
            learned_gap,
        )

    def forward_core(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        node_mask: torch.Tensor,
        baseline: torch.Tensor,
        learned_gap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the ONNX-compatible neural core with a precomputed gap gate."""
        valid = node_mask.to(dtype=features.dtype)
        spatial = self.coordinate_embedding(coordinates)
        input_embedding = self.input_projection(features)
        spatial_time = spatial[:, None, :, :].expand(-1, features.shape[1], -1, -1)
        representation = torch.cat([input_embedding, spatial_time], dim=-1)
        representation = representation * valid[:, None, :, None]
        for temporal, spatial_layer in zip(
            self.temporal_layers,
            self.spatial_layers,
            strict=True,
        ):
            representation = temporal(representation, valid)
            representation = spatial_layer(representation, spatial, valid)
        raw_residual = self.readout(representation).squeeze(-1)
        residual = torch.tanh(raw_residual) * self.config.max_residual
        residual = residual * learned_gap.to(residual.dtype) * valid[:, None, :]
        return baseline + residual, residual

    @staticmethod
    def _missing_run_lengths(missing: torch.Tensor) -> torch.Tensor:
        """Return total consecutive missing-run length at every position."""
        if missing.ndim != 3:
            raise ValueError("missing mask must have shape [batch, time, nodes]")
        left = torch.zeros_like(missing, dtype=torch.int64)
        right = torch.zeros_like(missing, dtype=torch.int64)
        for step in range(missing.shape[1]):
            if step:
                left[:, step] = torch.where(
                    missing[:, step],
                    left[:, step - 1] + 1,
                    0,
                )
            else:
                left[:, step] = missing[:, step].to(torch.int64)
        for step in range(missing.shape[1] - 1, -1, -1):
            if step + 1 < missing.shape[1]:
                right[:, step] = torch.where(
                    missing[:, step],
                    right[:, step + 1] + 1,
                    0,
                )
            else:
                right[:, step] = missing[:, step].to(torch.int64)
        return torch.where(missing, left + right - 1, 0)

    def get_config(self) -> dict[str, int | float]:
        """Return JSON/checkpoint-safe architecture parameters."""
        return asdict(self.config)


def temporal_spectral_regularization(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    visible_mask: torch.Tensor,
    node_mask: torch.Tensor,
    direction: bool = False,
) -> torch.Tensor:
    """Penalize temporal spectral magnitude, averaged over valid stations.

    Complete *all* unobserved entries with predictions, including natural gaps;
    only visible targets may enter this prior. FFTs never mix the unordered
    station axis, and padded stations contribute neither values nor weight.
    This is a temporal spectral prior, not the original 2-D low-rank penalty.

    Direction is normalized in half-turns. Its joint sine/cosine spectrum avoids
    a discontinuity at the wrap boundary and is invariant to the angular origin.
    """
    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError("prediction and target must have shape [batch, time, nodes]")
    if visible_mask.shape != prediction.shape:
        raise ValueError("visible_mask must match prediction shape")
    if node_mask.shape != (prediction.shape[0], prediction.shape[2]):
        raise ValueError("node_mask must have shape [batch, nodes]")
    valid = node_mask.bool()
    completed = torch.where(visible_mask.bool(), target, prediction)
    # Use where, not multiplication: padded placeholders may be NaN.
    completed = torch.where(valid[:, None, :], completed, 0.0)
    if direction:
        phase = torch.pi * completed
        completed = torch.stack((phase.sin(), phase.cos()), dim=-1)
    spectrum = torch.fft.rfft(completed, dim=1, norm="ortho")
    magnitude = (
        torch.linalg.vector_norm(spectrum, dim=-1) if direction else spectrum.abs()
    )
    per_station = magnitude.mean(dim=1).masked_fill(~valid, 0.0)
    return per_station.sum() / valid.sum().clamp_min(1)
