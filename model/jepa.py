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
from model.predictor import CFPredictor, Predictor
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


# ─── CF-JEPA ──────────────────────────────────────────────────────────────────

@dataclass
class CFJEPAConfig:
    """Config for CF-JEPA (mask-free, multi-horizon).

    Three horizon lengths replace the single n_patches_target:
      short = 1 patch  (~1 trading month)
      mid   = 2 patches (~2 months)
      long  = 3 patches (~3 months)

    crop_jitter_patches: context start is randomly shifted by up to this many
    patches within each base window, augmenting temporal diversity without masking.
    """
    n_features: int
    patch_len: int = 21
    n_patches_context: int = 9
    n_patches_short: int = 1
    n_patches_mid: int = 2
    n_patches_long: int = 3
    crop_jitter_patches: int = 1
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
    # Per-horizon loss weights (short / mid / long)
    w_short: float = 1.0
    w_mid: float = 1.0
    w_long: float = 1.0


class CFJEPA(nn.Module):
    """CF-JEPA: mask-free Joint Embedding Predictive Architecture.

    Key differences from JEPA:
    - Context is a randomly jittered temporal crop (no mask tokens)
    - CFPredictor predicts at three horizons using learned horizon embeddings
    - Online encoder (self.encoder) is preferred for classification / linear probe
    - EMA target encoder (self.target_encoder) is preferred for forecasting /
      anomaly detection — it produces smoother, lower-rank representations

    Training step expects a batch dict with keys:
        context, target_short, target_mid, target_long  — all [B, T, D]
    """

    def __init__(self, cfg: CFJEPAConfig):
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
        self.predictor = CFPredictor(
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
        context: torch.Tensor,       # [B, T_ctx,   D]
        target_short: torch.Tensor,  # [B, T_short, D]
        target_mid: torch.Tensor,    # [B, T_mid,   D]
        target_long: torch.Tensor,   # [B, T_long,  D]
    ) -> tuple[torch.Tensor, ...]:
        """Return (z_pred_short, z_pred_mid, z_pred_long,
                   z_tgt_short,  z_tgt_mid,  z_tgt_long)."""
        z_ctx = self.encoder(context)  # [B, N_ctx, d_model]

        z_pred_short = self.predictor(z_ctx, self.cfg.n_patches_short, horizon_id=0)
        z_pred_mid   = self.predictor(z_ctx, self.cfg.n_patches_mid,   horizon_id=1)
        z_pred_long  = self.predictor(z_ctx, self.cfg.n_patches_long,  horizon_id=2)

        with torch.no_grad():
            z_tgt_short = self.target_encoder(target_short)
            z_tgt_mid   = self.target_encoder(target_mid)
            z_tgt_long  = self.target_encoder(target_long)

        return z_pred_short, z_pred_mid, z_pred_long, z_tgt_short, z_tgt_mid, z_tgt_long

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted multi-horizon VICReg loss."""
        outputs = self.forward(
            batch["context"],
            batch["target_short"],
            batch["target_mid"],
            batch["target_long"],
        )
        z_pred_s, z_pred_m, z_pred_l, z_tgt_s, z_tgt_m, z_tgt_l = outputs

        kw = dict(lambda_inv=self.cfg.lambda_inv,
                  lambda_var=self.cfg.lambda_var,
                  lambda_cov=self.cfg.lambda_cov)
        loss_s, _ = vicreg_loss(z_pred_s, z_tgt_s.detach(), **kw)
        loss_m, _ = vicreg_loss(z_pred_m, z_tgt_m.detach(), **kw)
        loss_l, ms = vicreg_loss(z_pred_l, z_tgt_l.detach(), **kw)

        total = self.cfg.w_short * loss_s + self.cfg.w_mid * loss_m + self.cfg.w_long * loss_l

        metrics = {
            "loss_short": loss_s.item(),
            "loss_mid":   loss_m.item(),
            "loss_long":  loss_l.item(),
            "loss_total": total.item(),
            # Expose VICReg components from the long-horizon term for monitoring
            "loss_inv":   ms["loss_inv"],
            "loss_var":   ms["loss_var"],
            "loss_cov":   ms["loss_cov"],
        }
        return total, metrics

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Online encoder — discriminative, higher-rank. Use for linear probe (Exp 1)."""
        return self.encoder(x)

    @torch.no_grad()
    def encode_ema(self, x: torch.Tensor) -> torch.Tensor:
        """EMA target encoder — smoother, lower-rank. Use for forecasting / anomaly."""
        return self.target_encoder(x)

    def update_target_encoder(self) -> None:
        """Call after each optimizer step."""
        self.target_encoder.update(self.encoder)
