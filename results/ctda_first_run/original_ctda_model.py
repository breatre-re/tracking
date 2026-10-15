"""CTDA, with ambiguous architectural choices fixed and recorded for run 1."""
import torch
from torch import nn


class FeedForward(nn.Sequential):
    def __init__(self, dimension=40, hidden=160, dropout=.3):
        super().__init__(nn.Linear(dimension, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, dimension), nn.Dropout(dropout))


class ElementMix(nn.Module):
    def __init__(self, channels, tanh=True):
        super().__init__()
        self.linear = nn.Linear(channels, 1)
        self.tanh = tanh

    def forward(self, *features):
        mixed = self.linear(torch.stack(features, dim=-1)).squeeze(-1)
        return torch.tanh(mixed) if self.tanh else mixed


class RowColumnDecoderLayer(nn.Module):
    """Equations 12/13 in each branch's native orientation.

    Self attention uses own stream. Cross Q is FFN(other encoder).T;
    K/V and residual use the self-attention stream, as printed in Eq12.
    """
    def __init__(self, dimension=40, heads=4, hidden=160, dropout=.3):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.query_projection = FeedForward(dimension, hidden, dropout)
        self.ffn = FeedForward(dimension, hidden, dropout)
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(dimension) for _ in range(3)]
        self.dropout = nn.Dropout(dropout)

    def forward(self, own, other_native_encoder):
        attended = self.self_attention(own, own, own, need_weights=False)[0]
        first = self.norm1(own + self.dropout(attended))
        query = self.query_projection(other_native_encoder).transpose(-1, -2)
        attended = self.cross_attention(query, first, first, need_weights=False)[0]
        second = self.norm2(first + self.dropout(attended))
        return self.norm3(second + self.ffn(second))


def encoder(dimension, heads, hidden, dropout, layers):
    layer = nn.TransformerEncoderLayer(dimension, heads, hidden, dropout,
                                       activation='relu', batch_first=True, norm_first=False)
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


class RowColumnChannel(nn.Module):
    def __init__(self, dimension, heads, hidden, dropout, layers):
        super().__init__()
        self.column_encoder = encoder(dimension, heads, hidden, dropout, layers)
        self.row_encoder = encoder(dimension, heads, hidden, dropout, layers)
        self.column_decoders = nn.ModuleList([RowColumnDecoderLayer(dimension, heads, hidden, dropout) for _ in range(layers)])
        self.row_decoders = nn.ModuleList([RowColumnDecoderLayer(dimension, heads, hidden, dropout) for _ in range(layers)])
        self.mix = ElementMix(2)

    def forward(self, matrix):
        column_encoded = self.column_encoder(matrix)
        row_encoded = self.row_encoder(matrix.transpose(-1, -2))
        column, row = column_encoded, row_encoded
        for decoder in self.column_decoders: column = decoder(column, row_encoded)
        for decoder in self.row_decoders: row = decoder(row, column_encoded)
        return self.mix(column, row.transpose(-1, -2))


class ChannelDecoder(nn.Module):
    """Equation17: residual for first self MHA only, as printed."""
    def __init__(self, dimension, heads, hidden, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.query_mix = ElementMix(2)
        self.ffn = FeedForward(dimension, hidden, dropout)
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(dimension) for _ in range(3)]
        self.dropout = nn.Dropout(dropout)

    def forward(self, own, other1, other2):
        first = self.norm1(own + self.dropout(self.self_attention(own, own, own, need_weights=False)[0]))
        query = self.query_mix(other1, other2)
        second = self.norm2(self.dropout(self.cross_attention(query, first, first, need_weights=False)[0]))
        return self.norm3(self.ffn(second))


class CTDA(nn.Module):
    def __init__(self, dimension=40, heads=4, hidden=160, dropout=.3, row_column_layers=2):
        super().__init__()
        self.config = dict(dimension=dimension, heads=heads, hidden=hidden,
                           dropout=dropout, row_column_layers=row_column_layers)
        self.spatial_channels = nn.ModuleList([RowColumnChannel(dimension, heads, hidden, dropout, row_column_layers) for _ in range(3)])
        self.channel_encoders = nn.ModuleList([encoder(dimension, heads, hidden, dropout, 1) for _ in range(3)])
        self.channel_decoders = nn.ModuleList([ChannelDecoder(dimension, heads, hidden, dropout) for _ in range(3)])
        self.output_mix = ElementMix(3, tanh=False)

    def forward(self, inputs):
        encoded = [self.channel_encoders[i](self.spatial_channels[i](inputs[:, i])) for i in range(3)]
        decoded = [self.channel_decoders[i](encoded[i], encoded[(i + 1) % 3], encoded[(i + 2) % 3]) for i in range(3)]
        return self.output_mix(*decoded)  # Logits; sigmoid is applied for inference.


def focal_loss_elements(logits, targets, alpha=.75, gamma=2.):
    probability = torch.sigmoid(logits)
    pt = torch.where(targets.bool(), probability, 1 - probability)
    weight = torch.where(targets.bool(), alpha, 1 - alpha)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return weight * (1 - pt).pow(gamma) * bce
