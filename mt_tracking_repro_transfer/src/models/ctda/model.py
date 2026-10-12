"""Phase 4 CTDA network variants.

Paper source: Section 3.3, Eq. (11)-(18), p. 5-6.

Three variants share dimensions/heads/dropout/padding:

* ``standard_transformer`` - ablation baseline: one standard Transformer encoder
  over the flattened pairwise cell tokens (no row/column cross structure).
* ``row_column``           - row-column cross Transformer (Eq. 11-14), the
  element-wise mixing variant.
* ``row_column_3ch``       - the above plus the 3-channel cross Transformer fusion
  (Eq. 16-18) used by the paper's final model.

Reproduction assumptions (not printed in the paper) are marked in
``configs/phase4_ctda.yaml``: layer counts, FFN width, the exact per-cell
embedding for the standard baseline, and the batching of the padded sub-matrix
partition. The network here processes the padded ``P x P`` matrix directly (P =
padded input/output dimension 40) instead of the paper's sub-matrix batching,
which is under-specified for our data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CTDAConfig:
    variant: str = "row_column_3ch"
    pad_size: int = 40
    channels: int = 3
    d_model: int = 40
    n_heads: int = 4
    dropout: float = 0.3
    enc_layers: int = 2
    dec_layers: int = 2
    ffn_dim: int = 80

    @classmethod
    def from_dict(cls, payload: dict) -> "CTDAConfig":
        model = payload.get("model", {})
        features = payload.get("features", {})
        return cls(
            variant=str(model.get("variant", "row_column_3ch")),
            pad_size=int(payload.get("dataset", {}).get("pad_size", 40)),
            channels=len(features.get("channels", ["x_rdf", "y_rdf", "range_rdf"])),
            d_model=int(model.get("d_model", 40)),
            n_heads=int(model.get("n_heads", 4)),
            dropout=float(model.get("dropout", 0.3)),
            enc_layers=int(model.get("enc_layers", 2)),
            dec_layers=int(model.get("dec_layers", 2)),
            ffn_dim=int(model.get("ffn_dim", 80)),
        )


def _safe_mask(padding_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Never mask out every key for a query.

    Some frames have no valid candidate rows (or columns), which would otherwise
    make softmax attend over an all-masked sequence and return NaN. Keeping one
    (zeroed) key is harmless because its output is masked out afterwards.
    """

    if padding_mask is None:
        return None
    all_masked = padding_mask.all(dim=-1, keepdim=True)
    return padding_mask & ~all_masked


def _token_encoder(d_model: int, n_heads: int, ffn_dim: int, dropout: float, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        batch_first=True,
        activation="relu",
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


class RowColumnBlock(nn.Module):
    """Row-column cross Transformer for a single RDF channel (Eq. 11-14)."""

    def __init__(self, config: CTDAConfig) -> None:
        super().__init__()
        self.config = config
        self.column_encoder = _token_encoder(
            config.d_model, config.n_heads, config.ffn_dim, config.dropout, config.enc_layers
        )
        self.row_encoder = _token_encoder(
            config.d_model, config.n_heads, config.ffn_dim, config.dropout, config.enc_layers
        )
        self.column_decoder = nn.ModuleList(
            [
                nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout, batch_first=True)
                for _ in range(config.dec_layers)
            ]
        )
        self.row_decoder = nn.ModuleList(
            [
                nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout, batch_first=True)
                for _ in range(config.dec_layers)
            ]
        )
        self.mix = nn.Conv2d(2, 1, kernel_size=1)      # Eq. (14) element-wise mixing
        self.column_norm = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(config.dec_layers)])
        self.row_norm = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(config.dec_layers)])
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)

    def _cross_decode(
        self,
        query_source: torch.Tensor,
        self_source: torch.Tensor,
        cross_source: torch.Tensor,
        self_mask: Optional[torch.Tensor],
        cross_mask: Optional[torch.Tensor],
        layers: nn.ModuleList,
        norms: nn.ModuleList,
    ) -> torch.Tensor:
        """Two-MHA cross-fed decoder (Eq. 12-13) with optional padding masks."""

        hidden = query_source
        for index, attention in enumerate(layers):
            attended, _ = attention(
                hidden, self_source, self_source, key_padding_mask=self_mask, need_weights=False
            )
            hidden = norms[index](attended + hidden)
            attended, _ = attention(
                self.ffn(hidden),
                cross_source,
                cross_source,
                key_padding_mask=cross_mask,
                need_weights=False,
            )
            hidden = norms[index](attended + hidden)
            hidden = self.ffn_norm(self.ffn(hidden) + hidden)
        return hidden

    def forward(
        self,
        channel: torch.Tensor,
        row_mask: Optional[torch.Tensor] = None,
        col_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``channel`` is ``(B, P, P)`` with rows = candidates, columns = measurements."""

        row_padding = _safe_mask(None if row_mask is None else ~row_mask)
        col_padding = _safe_mask(None if col_mask is None else ~col_mask)

        column_hidden = self.column_encoder(channel, src_key_padding_mask=row_padding)      # rows as tokens
        row_hidden = self.row_encoder(channel.transpose(1, 2), src_key_padding_mask=col_padding).transpose(1, 2)

        column_decoded = self._cross_decode(
            column_hidden, column_hidden, row_hidden, row_padding, col_padding, self.column_decoder, self.column_norm
        )
        row_decoded = self._cross_decode(
            row_hidden.transpose(1, 2),
            row_hidden.transpose(1, 2),
            column_hidden.transpose(1, 2),
            col_padding,
            row_padding,
            self.row_decoder,
            self.row_norm,
        ).transpose(1, 2)
        stacked = torch.stack((column_decoded, row_decoded), dim=1)  # (B, 2, P, P)
        return torch.tanh(self.mix(stacked).squeeze(1))              # (B, P, P)


class CTDANet(nn.Module):
    """Association network producing per-cell logits over a ``P x P`` matrix."""

    def __init__(self, config: CTDAConfig) -> None:
        super().__init__()
        self.config = config
        self.pad_size = config.pad_size
        self.channels = config.channels
        self.variant = config.variant

        # Row-column variants treat a matrix row as a token of dimension pad_size,
        # so the model width must equal the padded dimension.
        if self.variant in ("row_column", "row_column_3ch") and config.d_model != config.pad_size:
            raise ValueError("row-column variants require d_model == pad_size")

        if self.variant == "standard_transformer":
            # Ablation baseline: a single standard Transformer encoder over row
            # and column tokens (2P tokens) with a dot-product association head.
            # No row-column cross structure and no positional encoding.
            self.token_projection = nn.Linear(config.channels * config.pad_size, config.d_model)
            self.segment_embedding = nn.Parameter(torch.zeros(2, config.d_model))
            self.encoder = _token_encoder(
                config.d_model, config.n_heads, config.ffn_dim, config.dropout, config.enc_layers
            )
            self.head_scale = nn.Parameter(torch.tensor(float(config.d_model) ** 0.5))
        elif self.variant == "row_column":
            self.blocks = nn.ModuleList([RowColumnBlock(config) for _ in range(config.channels)])
            self.head = nn.Conv2d(config.channels, 1, kernel_size=1)
        elif self.variant == "row_column_3ch":
            self.blocks = nn.ModuleList([RowColumnBlock(config) for _ in range(config.channels)])
            self.channel_encoders = nn.ModuleList(
                [
                    _token_encoder(config.d_model, config.n_heads, config.ffn_dim, config.dropout, config.enc_layers)
                    for _ in range(config.channels)
                ]
            )
            self.element_mix = nn.Conv2d(max(config.channels - 1, 1), 1, kernel_size=1)
            self.channel_self_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout, batch_first=True)
                    for _ in range(config.channels)
                ]
            )
            self.channel_decoders = nn.ModuleList(
                [
                    nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout, batch_first=True)
                    for _ in range(config.channels)
                ]
            )
            self.channel_self_norms = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(config.channels)])
            self.channel_norms = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(config.channels)])
            self.channel_ffns = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(config.d_model, config.ffn_dim),
                        nn.ReLU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.ffn_dim, config.d_model),
                    )
                    for _ in range(config.channels)
                ]
            )
            self.channel_ffn_norms = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(config.channels)])
            self.head = nn.Conv2d(config.channels, 1, kernel_size=1)   # Eq. (18) W_{3x1}
        else:
            raise ValueError(f"unknown CTDA variant {self.variant!r}")

    def forward(
        self,
        features: torch.Tensor,
        row_mask: Optional[torch.Tensor] = None,
        col_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-cell logits ``(B, P, P)``; masked cells are set to a large negative value."""

        batch = features.shape[0]
        pad = self.pad_size

        # Zero out padded cells so their values can never influence valid outputs;
        # combined with the key padding masks this makes the mask contract exact.
        if row_mask is not None and col_mask is not None:
            valid = (row_mask[:, :, None] & col_mask[:, None, :]).unsqueeze(1)
            features = features * valid.to(features.dtype)

        if self.variant == "standard_transformer":
            row_tokens = features.permute(0, 2, 1, 3).reshape(batch, pad, self.channels * pad)
            col_tokens = features.transpose(2, 3).permute(0, 2, 1, 3).reshape(batch, pad, self.channels * pad)
            tokens = torch.cat((row_tokens, col_tokens), dim=1)                  # (B, 2P, C*P)
            embedded = self.token_projection(tokens)
            embedded = embedded + torch.cat(
                (self.segment_embedding[0].expand(batch, pad, -1), self.segment_embedding[1].expand(batch, pad, -1)), dim=1
            )
            if row_mask is not None and col_mask is not None:
                valid_tokens = torch.cat((row_mask, col_mask), dim=1)
                padding_mask = _safe_mask(~valid_tokens)
            else:
                padding_mask = None
            hidden = self.encoder(embedded, src_key_padding_mask=padding_mask)
            row_hidden, col_hidden = hidden[:, :pad], hidden[:, pad:]
            logits = (row_hidden @ col_hidden.transpose(1, 2)) / self.head_scale.clamp(min=1e-3)
            return self._apply_mask(logits, row_mask, col_mask)

        channel_outputs = []
        for index in range(self.channels):
            channel_outputs.append(self.blocks[index](features[:, index], row_mask, col_mask))

        if self.variant == "row_column":
            cleaned = []
            for index, output in enumerate(channel_outputs):
                cleaned.append(self._mask_channel(output, row_mask, col_mask))
            stacked = torch.stack(cleaned, dim=1)
            logits = self.head(stacked).squeeze(1)
            return self._apply_mask(logits, row_mask, col_mask)

        # 3-channel cross Transformer fusion (Eq. 16-18).
        # ``channel_outputs`` has shape (B, rows, columns).  The 3-channel
        # encoder below transposes it to (B, columns, rows), so its token axis
        # is the measurement/column axis.  Masking by row_mask both hid valid
        # keys and left all-padding column sequences to softmax over -inf.
        col_padding = _safe_mask(None if col_mask is None else ~col_mask)
        encoded = []
        for index in range(self.channels):
            token_input = self._mask_channel(channel_outputs[index], row_mask, col_mask).transpose(1, 2)
            encoded.append(self.channel_encoders[index](token_input, src_key_padding_mask=col_padding))

        fused = []
        for index in range(self.channels):
            others = [encoded[other] for other in range(self.channels) if other != index]
            query = self.element_mix(torch.stack(others, dim=1)).squeeze(1)      # (B, P, P) query map
            self_attended, _ = self.channel_self_attentions[index](
                encoded[index], encoded[index], encoded[index], key_padding_mask=col_padding, need_weights=False
            )
            decoded_self = self.channel_self_norms[index](self_attended + encoded[index])
            cross_attended, _ = self.channel_decoders[index](
                query, decoded_self, decoded_self, key_padding_mask=col_padding, need_weights=False
            )
            decoded_cross = self.channel_norms[index](cross_attended + query)
            fused.append(self.channel_ffn_norms[index](self.channel_ffns[index](decoded_cross)))
        fused_stack = torch.stack(fused, dim=1)                                  # (B, C, P, P)
        logits = self.head(fused_stack).squeeze(1)
        return self._apply_mask(logits, row_mask, col_mask)

    @staticmethod
    def _mask_channel(channel: torch.Tensor, row_mask: Optional[torch.Tensor], col_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if row_mask is None or col_mask is None:
            return channel
        mask = row_mask[:, :, None] & col_mask[:, None, :]
        return channel * mask

    def _apply_mask(
        self, logits: torch.Tensor, row_mask: Optional[torch.Tensor], col_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if row_mask is None or col_mask is None:
            return logits
        mask = row_mask[:, :, None] & col_mask[:, None, :]
        return logits.masked_fill(~mask, -1e9)
