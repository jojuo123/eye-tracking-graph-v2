"""Cross-attention block: unlike `modules.transformer.TransformerEncoderBlock` (self-attention,
query = key = value), here the query sequence and the key/value ("context") sequence are
different token sets. Used to let per-fixation tokens gather information from an image's
spatial feature map at that fixation's location instead of exchanging information with each
other -- attending only into a fixed context keeps a query's output independent of the other
queries' identity or order, which is exactly what a permutation-invariant per-element encoding
needs (see `modules.multiscale_fixation_encoder`).
"""

import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + feed-forward (same structure as `TransformerEncoderBlock`,
    but attention keys/values come from a separate `context` sequence instead of the query
    sequence itself). The feed-forward step always has a residual; the attention step's
    residual is controlled by `residual` (default on, matching a standard transformer
    block) -- turned off, a level's cross-attention output *replaces* the incoming query
    instead of updating it, so nothing from before that level survives except through the
    attention itself.
    """

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, residual=True):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.residual = residual
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_context = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, context_key_padding_mask=None):
        """
        Args:
            query: `(B, N_q, d_model)`.
            context: `(B, N_kv, d_model)`, the keys/values attended into.
            context_key_padding_mask: optional `(B, N_kv)` bool tensor, True at positions of
                `context` that are padding and should be excluded from attention.
        """
        q = self.norm_query(query)
        kv = self.norm_context(context)
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=context_key_padding_mask, need_weights=False)
        attn_out = self.dropout(attn_out)
        query = query + attn_out if self.residual else attn_out
        query = query + self.ff(self.norm_ff(query))
        return query
