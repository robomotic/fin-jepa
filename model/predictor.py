"""
JEPA Predictor P_φ.

Takes the context encoder's output patches and predicts the target encoder's
latent representation for the masked (future) patches.

Architecture: narrow Transformer (smaller than encoder to avoid collapse).
The predictor must be expressive enough to map context → target latents,
but not so powerful that it shortcut-memorises without the encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.encoder import SinusoidalPositionalEncoding


class Predictor(nn.Module):
    """Predict target patch latents from context patch latents.

    The predictor receives:
      - context_latents: [B, N_ctx, d_model]  — from context encoder
      - n_target_patches: int                  — how many future patches to predict

    It outputs: [B, N_tgt, d_model]  — predicted latents for target patches.

    Target patch positions are represented by learned mask tokens + positional
    embeddings, following the I-JEPA / V-JEPA design pattern.
    """

    def __init__(
        self,
        d_model: int = 256,
        predictor_d_model: int = 128,   # narrower than encoder
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_patches: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.predictor_d_model = predictor_d_model

        # Project encoder d_model → predictor d_model
        self.input_proj = nn.Linear(d_model, predictor_d_model)

        # Learnable mask token (one per target patch position)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pos_enc = SinusoidalPositionalEncoding(
            predictor_d_model, max_len=max_patches, dropout=dropout
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=predictor_d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(predictor_d_model)

        # Project back to encoder d_model for loss computation
        self.output_proj = nn.Linear(predictor_d_model, d_model)

    def forward(
        self,
        context_latents: torch.Tensor,    # [B, N_ctx, d_model]
        n_target: int,
    ) -> torch.Tensor:                    # [B, N_tgt, d_model]
        B, N_ctx, _ = context_latents.shape

        # Project context to predictor dim
        memory = self.input_proj(context_latents)   # [B, N_ctx, pred_d]

        # Expand mask token to [B, N_tgt, pred_d] and add positional offsets
        tgt = self.mask_token.expand(B, n_target, -1)   # [B, N_tgt, pred_d]

        # Positional encoding: offset by N_ctx so target positions are distinct
        # from context positions in the PE table
        all_pos = self.pos_enc.pe[0, N_ctx : N_ctx + n_target]   # [N_tgt, pred_d]
        tgt = tgt + all_pos.unsqueeze(0)

        # Cross-attention: target queries, context keys/values
        out = self.transformer(tgt, memory)   # [B, N_tgt, pred_d]
        out = self.norm(out)
        out = self.output_proj(out)           # [B, N_tgt, d_model]
        return out


class CFPredictor(nn.Module):
    """Multi-horizon predictor for CF-JEPA (mask-free).

    Replaces learned mask tokens with per-horizon embeddings. The predictor
    receives context latents and a horizon_id (0=short, 1=mid, 2=long), and
    predicts target patch latents using cross-attention over the context.

    Eliminating the single shared mask token forces the model to rely on
    temporal position and horizon identity rather than a generic placeholder.
    """

    N_HORIZONS = 3  # short / mid / long

    def __init__(
        self,
        d_model: int = 256,
        predictor_d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_patches: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.predictor_d_model = predictor_d_model

        self.input_proj = nn.Linear(d_model, predictor_d_model)

        # One embedding per horizon type — replaces the single mask token
        self.horizon_embed = nn.Embedding(self.N_HORIZONS, predictor_d_model)

        self.pos_enc = SinusoidalPositionalEncoding(
            predictor_d_model, max_len=max_patches, dropout=dropout
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=predictor_d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(predictor_d_model)
        self.output_proj = nn.Linear(predictor_d_model, d_model)

    def forward(
        self,
        context_latents: torch.Tensor,  # [B, N_ctx, d_model]
        n_target: int,
        horizon_id: int,                # 0=short, 1=mid, 2=long
    ) -> torch.Tensor:                  # [B, N_tgt, d_model]
        B, N_ctx, _ = context_latents.shape
        device = context_latents.device

        memory = self.input_proj(context_latents)  # [B, N_ctx, pred_d]

        # Horizon embedding broadcast over target patches
        hid = torch.full((B, n_target), horizon_id, dtype=torch.long, device=device)
        h_embed = self.horizon_embed(hid)  # [B, N_tgt, pred_d]

        # Positional encoding offset by context length
        pos = self.pos_enc.pe[0, N_ctx : N_ctx + n_target]  # [N_tgt, pred_d]
        tgt = h_embed + pos.unsqueeze(0)

        out = self.transformer(tgt, memory)
        out = self.norm(out)
        return self.output_proj(out)  # [B, N_tgt, d_model]
