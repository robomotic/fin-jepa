"""
JEPA training module.

Loss: VICReg (Bardes et al. 2022) in latent space.
  L = λ_inv * invariance(z_pred, z_target)   ← make predictions match targets
    + λ_var * variance(z_pred)               ← prevent representation collapse
    + λ_cov * covariance(z_pred)             ← decorrelate embedding dimensions

VICReg eliminates the need for negative samples or stop-gradient tricks
(other than the EMA target encoder, which is structural, not a loss trick).

Training loop:
  1. Encode context with online encoder → z_ctx  [B, N_ctx, d_model]
  2. Encode target  with target encoder → z_tgt  [B, N_tgt, d_model]  (no grad)
  3. Predict target latents from context → z_pred [B, N_tgt, d_model]
  4. Compute VICReg(z_pred, z_tgt)
  5. Backprop through online encoder + predictor only
  6. EMA-update target encoder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import ContextEncoder
from model.predictor import Predictor
from model.target_encoder import TargetEncoder


# ─── VICReg Loss ──────────────────────────────────────────────────────────────

def vicreg_loss(
    z_pred: torch.Tensor,   # [B, N, d_model]
    z_target: torch.Tensor, # [B, N, d_model]  — no gradient should flow here
    lambda_inv: float = 25.0,
    lambda_var: float = 25.0,
    lambda_cov: float = 1.0,
    gamma: float = 1.0,     # target std for variance term
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute VICReg loss between predicted and target latents.

    Flattens the N-patch dimension so loss is computed over B×N vectors.
    """
    B, N, D = z_pred.shape
    z_pred   = z_pred.reshape(B * N, D)
    z_target = z_target.reshape(B * N, D)

    # Invariance: mean squared error between predictions and targets
    inv_loss = F.mse_loss(z_pred, z_target)

    # Variance: penalise if std of each dimension < gamma
    z_pred_centered   = z_pred   - z_pred.mean(dim=0, keepdim=True)
    z_target_centered = z_target - z_target.mean(dim=0, keepdim=True)

    std_pred   = torch.sqrt(z_pred_centered.var(dim=0) + eps)
    std_target = torch.sqrt(z_target_centered.var(dim=0) + eps)
    var_loss = (F.relu(gamma - std_pred).mean() + F.relu(gamma - std_target).mean()) / 2

    # Covariance: off-diagonal elements of covariance matrix should be ~0
    def cov_loss(z, z_c):
        n = z.shape[0]
        cov = (z_c.T @ z_c) / (n - 1)
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag / D

    cov = (cov_loss(z_pred, z_pred_centered) + cov_loss(z_target, z_target_centered)) / 2

    loss = lambda_inv * inv_loss + lambda_var * var_loss + lambda_cov * cov

    metrics = {
        "loss_inv":   inv_loss.item(),
        "loss_var":   var_loss.item(),
        "loss_cov":   cov.item(),
        "loss_total": loss.item(),
    }
    return loss, metrics


# ─── Full JEPA Model ──────────────────────────────────────────────────────────

@dataclass
class JEPAConfig:
    n_features: int
    patch_len: int = 21
    n_patches_context: int = 9
    n_patches_target: int = 3
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    predictor_d_model: int = 128
    n_predictor_layers: int = 4
    tau_start: float = 0.996
    tau_end: float = 0.9999
    lambda_inv: float = 25.0
    lambda_var: float = 25.0
    lambda_cov: float = 1.0


class JEPA(nn.Module):
    """Full JEPA: online encoder + predictor + EMA target encoder."""

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = ContextEncoder(
            n_features=cfg.n_features,
            patch_len=cfg.patch_len,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_encoder_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
        )
        self.predictor = Predictor(
            d_model=cfg.d_model,
            predictor_d_model=cfg.predictor_d_model,
            n_heads=cfg.n_heads // 2,
            n_layers=cfg.n_predictor_layers,
            d_ff=cfg.d_ff // 2,
            dropout=cfg.dropout,
        )
        self.target_encoder = TargetEncoder(
            online_encoder=self.encoder,
            tau_start=cfg.tau_start,
            tau_end=cfg.tau_end,
        )

    def forward(
        self,
        context: torch.Tensor,  # [B, T_ctx, D]
        target: torch.Tensor,   # [B, T_tgt, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Returns:
            z_pred:   [B, N_tgt, d_model]  — predictor output
            z_target: [B, N_tgt, d_model]  — target encoder output (no grad)
        """
        # Online encoder encodes context
        z_ctx = self.encoder(context)                            # [B, N_ctx, d_model]

        # Predict target patch latents
        z_pred = self.predictor(z_ctx, n_target=self.cfg.n_patches_target)  # [B, N_tgt, d_model]

        # Target encoder produces ground-truth latents (no gradient)
        with torch.no_grad():
            z_target = self.target_encoder(target)               # [B, N_tgt, d_model]

        return z_pred, z_target

    def training_step(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss and return metrics dict."""
        z_pred, z_target = self.forward(context, target)
        loss, metrics = vicreg_loss(
            z_pred, z_target.detach(),
            lambda_inv=self.cfg.lambda_inv,
            lambda_var=self.cfg.lambda_var,
            lambda_cov=self.cfg.lambda_cov,
        )
        return loss, metrics

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a window into latent patch representations.

        Used in experiments to extract frozen representations.
        """
        return self.encoder(x)

    def update_target_encoder(self) -> None:
        """Call after each optimizer step."""
        self.target_encoder.update(self.encoder)
