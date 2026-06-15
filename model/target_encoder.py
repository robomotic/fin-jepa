"""
EMA (Exponential Moving Average) target encoder.

Maintains a momentum copy of the context encoder whose weights are NEVER
gradient-updated — only updated via EMA after each optimizer step.

Update rule:  θ_ema ← τ * θ_ema + (1 - τ) * θ_online
τ is annealed from tau_start → tau_end over training via cosine schedule.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from model.encoder import ContextEncoder


class TargetEncoder(nn.Module):
    def __init__(
        self,
        online_encoder: ContextEncoder,
        tau_start: float = 0.996,
        tau_end: float = 0.9999,
        total_steps: int = 100_000,
    ):
        super().__init__()
        self.encoder = copy.deepcopy(online_encoder)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_steps = total_steps
        self._step = 0

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        """EMA update. Call once per optimizer step."""
        tau = self._get_tau()
        for p_ema, p_online in zip(self.encoder.parameters(), online_encoder.parameters()):
            p_ema.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)
        self._step += 1

    def _get_tau(self) -> float:
        progress = min(self._step / max(self.total_steps, 1), 1.0)
        # Cosine anneal from tau_start to tau_end
        return self.tau_end - (self.tau_end - self.tau_start) * (
            1 + math.cos(math.pi * progress)
        ) / 2

    @property
    def current_tau(self) -> float:
        return self._get_tau()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encoder(x, **kwargs)
