"""A minimal, dependency-free (beyond torch) pre-norm Transformer encoder."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer encoder block: self-attention + feed-forward, each with a residual."""

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """A stack of pre-norm `TransformerEncoderBlock`s with sinusoidal positional encoding.

    Set `use_positional_encoding=False` for inputs whose *order* carries no meaning of its
    own -- e.g. a set of per-fixation embeddings in a permutation-learning task, where the
    sequence order is exactly the shuffled arrangement a downstream head has to recover.
    `PositionalEncoding` is keyed by index, so leaving it on would hand the model a direct
    readout of that shuffled order, contaminating the very thing it's supposed to predict.
    With it off, self-attention remains permutation-equivariant: permuting the input
    permutes the output identically, since nothing here depends on position, only content
    and (via `key_padding_mask`) which tokens are valid.
    """

    def __init__(self, d_model, n_heads, n_layers, d_ff=None, dropout=0.1, max_len=5000, use_positional_encoding=True):
        super().__init__()
        self.use_positional_encoding = use_positional_encoding
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout) if use_positional_encoding else None
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.norm(x)
