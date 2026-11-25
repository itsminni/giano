"""Temporal, cross-variable and geographic attention for the retained Giano ensemble."""

from __future__ import annotations

import torch
from torch import nn

VARIABLES = (
    "temperature",
    "humidity",
    "pressure",
    "precipitation",
    "wind_speed",
    "wind_direction",
)


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


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.variable_encoders = nn.ModuleList(nn.Linear(20, 8) for _ in VARIABLES)
        self.coordinates = CoordinateEmbedding(8, 2)
        self.temporal = ProjectedTemporalAttention(16, 2, 4, 32, 0.0)
        self.spatial = EmbeddedSpatialAttention(16, 8, 32, 0.0)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 1))
            for _ in VARIABLES
        )
        self.variable_attention = nn.MultiheadAttention(
            16, 2, dropout=0.0, batch_first=True
        )
        self.variable_norm_attention = nn.LayerNorm(16)
        self.variable_norm_feed_forward = nn.LayerNorm(16)
        self.variable_feed_forward = nn.Sequential(
            nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16)
        )


INPUTS = (
    "values",
    "input_mask",
    "availability",
    "baseline",
    "coordinates",
    "node_mask",
    "auxiliary",
    "auxiliary_mask",
    "center",
    "scale",
    "reference_valid",
    "baseline_source",
    "donor_count",
    "calendar",
)


class LongGapModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()

    def encode(
        self, features: torch.Tensor, coordinates: torch.Tensor, nodes: torch.Tensor
    ) -> torch.Tensor:
        embedding = torch.stack(
            [
                encoder(features[..., j, :])
                for j, encoder in enumerate(self.backbone.variable_encoders)
            ],
            dim=-2,
        )
        batch, steps, count, variables, _ = embedding.shape
        spatial = self.backbone.coordinates(coordinates)
        representation = torch.cat(
            [
                embedding,
                spatial[:, None, :, None, :].expand(-1, steps, -1, variables, -1),
            ],
            dim=-1,
        )
        representation = torch.where(nodes[:, None, :, None, None], representation, 0.0)
        station_mask = (
            nodes[:, None, :]
            .expand(-1, variables, -1)
            .reshape(batch * variables, count)
        )

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(0, 3, 1, 2, 4).reshape(
                batch * variables, steps, count, 16
            )

        def unpack(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, variables, steps, count, 16).permute(
                0, 2, 3, 1, 4
            )

        representation = unpack(
            self.backbone.temporal(pack(representation), station_mask)
        )
        sequence = representation.reshape(batch * steps * count, variables, 16)
        # Every real station has six typed slots, including unavailable-variable
        # slots. Their sanitized inputs carry absence, while node padding stays 0.
        message, _ = self.backbone.variable_attention(
            sequence, sequence, sequence, need_weights=False
        )
        sequence = self.backbone.variable_norm_attention(sequence + message)
        sequence = self.backbone.variable_norm_feed_forward(
            sequence + self.backbone.variable_feed_forward(sequence)
        )
        representation = sequence.reshape(batch, steps, count, variables, 16)
        representation = torch.where(nodes[:, None, :, None, None], representation, 0.0)
        spatial_tokens = (
            spatial[:, None]
            .expand(-1, variables, -1, -1)
            .reshape(batch * variables, count, 8)
        )
        representation = unpack(
            self.backbone.spatial(pack(representation), spatial_tokens, station_mask)
        )
        return torch.cat(
            [
                head(representation[..., j, :])
                for j, head in enumerate(self.backbone.heads)
            ],
            dim=-1,
        )

    def forward(
        self,
        values: torch.Tensor,
        input_mask: torch.Tensor,
        availability: torch.Tensor,
        baseline: torch.Tensor,
        coordinates: torch.Tensor,
        node_mask: torch.Tensor,
        auxiliary: torch.Tensor,
        auxiliary_mask: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        reference_valid: torch.Tensor,
        baseline_source: torch.Tensor,
        donor_count: torch.Tensor,
        calendar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            values.ndim != 4
            or values.shape[-1] != 6
            or any(
                t.shape != values.shape
                for t in (
                    input_mask,
                    availability,
                    baseline,
                    auxiliary,
                    auxiliary_mask,
                    baseline_source,
                    donor_count,
                )
            )
        ):
            raise ValueError(
                "Long-gap inputs require matching batch/time/node/six-variable tensors"
            )
        b, t, n, _ = values.shape
        if (
            node_mask.shape != (b, n)
            or coordinates.shape != (b, n, 2)
            or center.shape != (b, n, 6)
            or scale.shape != center.shape
            or reference_valid.shape != center.shape
            or calendar.shape != (b, t, 4)
        ):
            raise ValueError("Invalid long-gap static/calendar geometry")
        if any(
            m.dtype != torch.bool
            for m in (
                input_mask,
                availability,
                auxiliary_mask,
                node_mask,
                reference_valid,
            )
        ):
            raise ValueError("Long-gap masks must be boolean")
        active = availability & node_mask[:, None, :, None]
        visible = input_mask & active
        external = auxiliary_mask & active
        clean = torch.where(visible, values, 0.0)
        base = torch.where(active, baseline, 0.0)
        aux = torch.where(external, auxiliary, 0.0)
        c = torch.where(node_mask[..., None], center, 0.0)
        s = torch.where(node_mask[..., None], scale, 1.0)
        coords = torch.where(node_mask[..., None], coordinates, 0.0)
        # Check the masked tensors directly. Boolean indexing would allocate a
        # variable-size tensor and synchronise the accelerator for each field.
        unsupported = (
            (input_mask | auxiliary_mask) & ~availability & node_mask[:, None, :, None]
        ).any()
        finite = torch.stack(
            [
                torch.isfinite(x).all()
                for x in (clean, aux, base, coords, c, s, calendar)
            ]
        ).all()
        if bool(unsupported | ~finite | (s <= 0).any()):
            if bool(unsupported):
                raise ValueError("Long-gap observations must be structurally available")
            raise ValueError("Nonfinite visible long-gap input or invalid scale")
        offsets = c.new_tensor([0.0, 50.0, 1000.0, 0.0, 0.0])
        units = c.new_tensor([20.0, 50.0, 50.0, 1.0, 5.0])
        ref1 = torch.cat(
            (
                torch.asinh((c[..., :5] - offsets) / units),
                torch.sin(c[..., 5:] * torch.pi / 180),
            ),
            dim=-1,
        )
        ref2 = torch.cat(
            (torch.asinh(s[..., :5] / units), torch.cos(c[..., 5:] * torch.pi / 180)),
            dim=-1,
        )

        # Only wind direction uses two circular channels. Build all six slots
        # together so each feature needs one operation across the whole batch.
        def observation_channels(data, mask):
            angle = torch.pi * data[..., 5:]
            direction = mask[..., 5:]
            return (
                torch.cat(
                    (data[..., :5], torch.where(direction, angle.sin(), 0.0)), dim=-1
                ),
                torch.cat(
                    (
                        torch.zeros_like(data[..., :5]),
                        torch.where(direction, angle.cos(), 0.0),
                    ),
                    dim=-1,
                ),
            )

        inputs = torch.stack(
            [
                *observation_channels(clean, visible),
                *observation_channels(base, active),
                *observation_channels(aux, external),
                visible.to(values.dtype),
                external.to(values.dtype),
                ref1[:, None].expand_as(values),
                ref2[:, None].expand_as(values),
                torch.where(active, donor_count.to(values.dtype) / 8, 0.0),
                (baseline_source == 1).to(values.dtype),
                (baseline_source == 2).to(values.dtype),
                (baseline_source == 3).to(values.dtype),
                *[calendar[..., k, None, None].expand_as(values) for k in range(4)],
                active.to(values.dtype),
                reference_valid[:, None].expand_as(values).to(values.dtype),
            ],
            dim=-1,
        )
        inputs = torch.where(node_mask[:, None, :, None, None], inputs, 0.0)
        raw = self.encode(inputs, coords, node_mask)
        residual = torch.where(active & ~visible, raw, 0.0)
        scalar = base[..., :5] + residual[..., :5]
        angle = torch.remainder(base[..., 5:] + residual[..., 5:] + 1.0, 2.0) - 1.0
        prediction = torch.cat((scalar, angle), dim=-1)
        prediction = torch.where(visible, clean, prediction)
        return torch.where(active, prediction, 0.0), residual
