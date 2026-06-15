"""
Baseline encoders for Experiment 1 (linear probing).

Three baselines against which JEPA's linear probe IC is compared:

1. Random encoder     — randomly initialised, never trained; establishes floor
2. Raw features       — identity (pass raw inputs directly); tests whether JEPA
                        adds anything over simple compression
3. Shuffled-sequence  — JEPA trained with sequence order randomised per window;
                        isolates whether temporal structure is learned
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from model.encoder import ContextEncoder
from model.jepa import JEPA, JEPAConfig


def make_random_encoder(cfg: JEPAConfig) -> ContextEncoder:
    """Return a freshly-initialised (never trained) encoder."""
    enc = ContextEncoder(
        n_features=cfg.n_features,
        patch_len=cfg.patch_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_encoder_layers,
        d_ff=cfg.d_ff,
        dropout=0.0,  # no dropout during eval
    )
    # All parameters stay at PyTorch default init
    return enc


class RawFeaturesEncoder(nn.Module):
    """'Encoder' that simply mean-pools the context window patches.

    Produces [B, N_patches, D_in] → projects to d_model via linear.
    This tests whether a linear compression of raw features matches JEPA.
    """

    def __init__(self, n_features: int, patch_len: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(n_features * patch_len, d_model)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        Returns: [B, N_patches, d_model]
        """
        B, T, D = x.shape
        N = T // self.patch_len
        patches = x.reshape(B, N, self.patch_len * D)
        return self.proj(patches)


def make_shuffled_sequence_dataset(dataset, seed: int = 99) -> "ShuffledSequenceWrapper":
    """Wrap a FinancialJEPADataset so that each context window has its
    timesteps permuted. Trains JEPA to ignore temporal order.
    """
    return ShuffledSequenceWrapper(dataset, seed=seed)


class ShuffledSequenceWrapper:
    """Dataset wrapper that randomly permutes timesteps within each window."""

    def __init__(self, base_dataset, seed: int = 99):
        self.base = base_dataset
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        ctx = sample["context"].numpy().copy()   # [T_ctx, D]
        perm = self.rng.permutation(ctx.shape[0])
        ctx = ctx[perm]
        sample = dict(sample)
        sample["context"] = torch.from_numpy(ctx)
        return sample
