"""
Context encoder E_θ: maps an input window to a sequence of patch latents.

Architecture:
  1. Split [B, T, D] into [B, N_patches, patch_len × D] via PatchEmbed
  2. Linear project to d_model
  3. Sinusoidal positional encoding
  4. Transformer encoder (non-causal — each patch can see all context patches)
  5. Output: [B, N_patches, d_model]

Non-causal is appropriate here because the context window is fully observed.
Causal masking would only be needed for autoregressive decoding, which is not
the JEPA objective.

If TS-JEPA is cloned into extern/TS_JEPA/, this module can be swapped for
their Encoder with the same signature. See the shim at the bottom.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Split time series into non-overlapping patches and project to d_model."""

    def __init__(self, patch_len: int, n_features: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len * n_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]  — T must be divisible by patch_len
        Returns:
            [B, N, d_model]  where N = T // patch_len
        """
        B, T, D = x.shape
        assert T % self.patch_len == 0, f"T={T} must be divisible by patch_len={self.patch_len}"
        N = T // self.patch_len
        # Reshape: [B, N, patch_len * D]
        x = x.reshape(B, N, self.patch_len * D)
        return self.proj(x)  # [B, N, d_model]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ContextEncoder(nn.Module):
    """JEPA context encoder.

    Input:  [B, T_ctx, D_in]
    Output: [B, N_patches, d_model]   where N_patches = T_ctx // patch_len
    """

    def __init__(
        self,
        n_features: int,
        patch_len: int = 21,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_patches: int = 64,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model

        self.patch_embed = PatchEmbed(patch_len, n_features, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_patches, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,                    # [B, T, D]
        src_key_padding_mask: torch.Tensor | None = None,  # [B, N_patches] bool
    ) -> torch.Tensor:                       # [B, N_patches, d_model]
        z = self.patch_embed(x)              # [B, N, d_model]
        z = self.pos_enc(z)
        z = self.transformer(z, src_key_padding_mask=src_key_padding_mask)
        z = self.norm(z)
        return z


# ─── Optional TS-JEPA shim ───────────────────────────────────────────────────

def load_ts_jepa_encoder(
    n_features: int,
    patch_len: int = 21,
    d_model: int = 256,
    extern_path: str = "extern/TS_JEPA",
) -> nn.Module:
    """Try to load the TS-JEPA Encoder; fall back to ContextEncoder.

    Call this instead of ContextEncoder() to use the TS-JEPA backbone
    when extern/TS_JEPA has been cloned.
    """
    import importlib.util, sys
    from pathlib import Path

    src_path = Path(extern_path) / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        try:
            spec = importlib.util.spec_from_file_location("Encoder", src_path / "Encoder.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # TS-JEPA's Encoder class — wrap to match our signature
            return _TSJEPAEncoderWrapper(mod.Encoder, n_features, patch_len, d_model)
        except Exception as e:
            pass

    return ContextEncoder(n_features=n_features, patch_len=patch_len, d_model=d_model)


class _TSJEPAEncoderWrapper(nn.Module):
    """Thin wrapper that adapts TS-JEPA's Encoder to our (B, T, D) signature."""

    def __init__(self, EncoderClass, n_features, patch_len, d_model):
        super().__init__()
        # TS-JEPA Encoder expects (seq_len, patch_num, d_model) style config;
        # construct it with sensible defaults and adapt I/O shapes.
        self.encoder = EncoderClass(
            seq_len=patch_len,
            patch_num=None,       # inferred at runtime
            d_model=d_model,
            n_heads=8,
            e_layers=6,
            d_ff=d_model * 4,
            dropout=0.1,
            activation="gelu",
        )
        self.patch_len = patch_len
        self.proj = nn.Linear(n_features * patch_len, d_model)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, T, D = x.shape
        N = T // self.patch_len
        patches = x.reshape(B, N, self.patch_len * D)
        patches = self.proj(patches)
        return self.encoder(patches)
