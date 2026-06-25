"""
PyTorch sliding-window Dataset for JEPA training and evaluation.

Each sample produces:
  context: [T_ctx, D]  — input window (all features)
  target:  [T_tgt, D]  — future window (same features)
  mask:    [D]          — column mask (1=visible, 0=zeroed for context masking experiments)
  meta:    SampleMeta

Window layout:
  [──────── context (T_ctx steps) ────────][── target (T_tgt steps) ──]
  Total = T_ctx + T_tgt trading days

T_ctx = patch_len × n_patches_context  (default: 21 × 9 = 189)
T_tgt = patch_len × n_patches_target   (default: 21 × 3 = 63)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SampleMeta:
    start_date: str
    context_end_date: str
    target_end_date: str
    ffill_fraction: float


class FinancialJEPADataset(Dataset):
    """Sliding-window dataset that feeds the JEPA encoder-predictor pair.

    Masking strategies (for experiments):
      "none"         — all columns visible in context
      "pillar"       — zero out all columns belonging to specified pillars
      "random_col"   — randomly zero out masking_fraction of columns per sample
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        config: dict,
        patch_len: int = 21,
        n_patches_context: int = 9,
        n_patches_target: int = 3,
        stride: int = 5,
        ffill_mask: Optional[pd.DataFrame] = None,
        masking_strategy: str = "none",
        pillars_to_mask: Optional[list[int]] = None,
        masking_fraction: float = 0.0,
        seed: int = 42,
    ):
        self.panel = panel
        self.config = config
        self.patch_len = patch_len
        self.n_ctx = n_patches_context
        self.n_tgt = n_patches_target
        self.context_len = patch_len * n_patches_context
        self.target_len  = patch_len * n_patches_target
        self.window_len  = self.context_len + self.target_len
        self.stride = stride
        self.ffill_mask = ffill_mask
        self.masking_strategy = masking_strategy
        self.pillars_to_mask = pillars_to_mask or []
        self.masking_fraction = masking_fraction
        self.rng = np.random.default_rng(seed)

        self.columns = list(panel.columns)
        self.col_idx = {c: i for i, c in enumerate(self.columns)}
        self.D = len(self.columns)

        # Precompute pillar membership for masking
        self._pillar_col_mask = self._build_pillar_col_mask()

        # Precompute valid start indices
        self._valid_starts = self._build_valid_starts()

    def _build_pillar_col_mask(self) -> dict[int, np.ndarray]:
        """For each pillar, a bool array [D] where True = belongs to that pillar."""
        series_cfg = self.config.get("series", {})
        masks: dict[int, np.ndarray] = {}
        for pillar in range(1, 7):
            m = np.zeros(self.D, dtype=bool)
            for col in self.columns:
                cfg = series_cfg.get(col, {})
                if cfg.get("pillar") == pillar:
                    m[self.col_idx[col]] = True
            masks[pillar] = m
        return masks

    def _build_valid_starts(self) -> list[int]:
        """Precompute indices of all valid window start positions."""
        n = len(self.panel)
        if n < self.window_len:
            return []

        missing_threshold = self.config.get("normalization", {}).get(
            "missing_window_threshold", 0.20
        )
        valid = []
        for i in range(0, n - self.window_len + 1, self.stride):
            window = self.panel.iloc[i : i + self.window_len]
            nan_frac = window.isna().values.mean()
            if nan_frac <= missing_threshold:
                valid.append(i)

        return valid

    def __len__(self) -> int:
        return len(self._valid_starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | SampleMeta]:
        start = self._valid_starts[idx]
        window = self.panel.iloc[start : start + self.window_len]

        # Fill remaining NaNs (should be rare after pipeline forward-fill)
        values = window.values.astype(np.float32)
        nan_mask_arr = np.isnan(values)
        if nan_mask_arr.any():
            # Simple column-wise median fill for isolated NaNs.
            # Fall back to 0.0 for all-NaN columns (data is z-scored, so 0 = global mean).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                col_medians = np.nanmedian(values, axis=0)
            col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
            values = np.where(nan_mask_arr, col_medians[np.newaxis, :], values)

        context = torch.from_numpy(values[: self.context_len])   # [T_ctx, D]
        target  = torch.from_numpy(values[self.context_len :])   # [T_tgt, D]

        # Build column visibility mask [D]
        col_mask = self._build_col_mask()

        # Apply mask to context (zero-out invisible channels)
        context = context * col_mask.unsqueeze(0).float()

        # Compute forward-fill fraction for metadata
        ffill_frac = 0.0
        if self.ffill_mask is not None:
            ffill_window = self.ffill_mask.iloc[start : start + self.window_len]
            ffill_frac = float(ffill_window.values.mean())

        dates = window.index
        meta = SampleMeta(
            start_date=str(dates[0].date()),
            context_end_date=str(dates[self.context_len - 1].date()),
            target_end_date=str(dates[-1].date()),
            ffill_fraction=ffill_frac,
        )

        return {
            "context": context,   # [T_ctx, D]
            "target":  target,    # [T_tgt, D]
            "mask":    col_mask,  # [D]   bool
            "meta":    meta,
        }

    def _build_col_mask(self) -> torch.Tensor:
        """Build a [D] mask where 1 = visible, 0 = masked in context."""
        mask = torch.ones(self.D, dtype=torch.bool)

        if self.masking_strategy == "pillar":
            for p in self.pillars_to_mask:
                m = self._pillar_col_mask.get(p, np.zeros(self.D, dtype=bool))
                mask[torch.from_numpy(m)] = False

        elif self.masking_strategy == "random_col":
            n_mask = int(self.D * self.masking_fraction)
            if n_mask > 0:
                idxs = self.rng.choice(self.D, size=n_mask, replace=False)
                mask[idxs] = False

        return mask

    def get_pillar_mask(self, pillars_to_mask: list[int]) -> torch.Tensor:
        """Return a [D] bool mask with False for columns in specified pillars.

        Convenience method for experiments that need to construct masks externally.
        """
        mask = torch.ones(self.D, dtype=torch.bool)
        for p in pillars_to_mask:
            m = self._pillar_col_mask.get(p, np.zeros(self.D, dtype=bool))
            mask[torch.from_numpy(m)] = False
        return mask

    def get_equity_only_mask(self) -> torch.Tensor:
        """Mask that zeros all non-equity (non-yahoo) series.

        Used in the falsifiability check: can JEPA predict macro shocks from
        equity prices alone? (it should not be able to reliably).
        """
        series_cfg = self.config.get("series", {})
        mask = torch.zeros(self.D, dtype=torch.bool)
        for col in self.columns:
            if series_cfg.get(col, {}).get("source") == "yahoo":
                mask[self.col_idx[col]] = True
        return mask

    def get_macro_only_mask(self) -> torch.Tensor:
        """Mask that zeros all equity (yahoo) series, keeping only macro inputs."""
        return ~self.get_equity_only_mask()

    def get_pillar_only_mask(self, pillar: int) -> torch.Tensor:
        """Keep only columns from one specific pillar, zero everything else."""
        m = self._pillar_col_mask.get(pillar, np.zeros(self.D, dtype=bool))
        return torch.from_numpy(m)


class CFJEPADataset(Dataset):
    """Sliding-window dataset for CF-JEPA training.

    Each sample provides:
      context:      [T_ctx, D]   — randomly jittered contiguous crop
      target_short: [T_short, D] — next n_patches_short patches after context
      target_mid:   [T_mid,   D] — next n_patches_mid   patches after context
      target_long:  [T_long,  D] — next n_patches_long  patches after context

    The crop jitter (up to crop_jitter_patches × patch_len timesteps) shifts the
    context start within each base window, augmenting training diversity without
    needing a separate masking mechanism.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        config: dict,
        patch_len: int = 21,
        n_patches_context: int = 9,
        n_patches_short: int = 1,
        n_patches_mid: int = 2,
        n_patches_long: int = 3,
        crop_jitter_patches: int = 1,
        stride: int = 5,
        seed: int = 42,
    ):
        self.panel = panel
        self.config = config
        self.patch_len = patch_len
        self.n_ctx = n_patches_context
        self.n_short = n_patches_short
        self.n_mid = n_patches_mid
        self.n_long = n_patches_long

        self.context_len = patch_len * n_patches_context
        self.short_len   = patch_len * n_patches_short
        self.mid_len     = patch_len * n_patches_mid
        self.long_len    = patch_len * n_patches_long
        self.jitter      = crop_jitter_patches * patch_len  # timesteps

        # Total panel slice needed: jitter room + context + longest target
        self.window_len = self.jitter + self.context_len + self.long_len

        self.stride = stride
        self.rng = np.random.default_rng(seed)
        self.columns = list(panel.columns)
        self.D = len(self.columns)

        self._valid_starts = self._build_valid_starts()

    def _build_valid_starts(self) -> list[int]:
        n = len(self.panel)
        if n < self.window_len:
            return []
        missing_threshold = self.config.get("normalization", {}).get(
            "missing_window_threshold", 0.20
        )
        valid = []
        for i in range(0, n - self.window_len + 1, self.stride):
            window = self.panel.iloc[i : i + self.window_len]
            if window.isna().values.mean() <= missing_threshold:
                valid.append(i)
        return valid

    def __len__(self) -> int:
        return len(self._valid_starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base = self._valid_starts[idx]
        window = self.panel.iloc[base : base + self.window_len]

        values = window.values.astype(np.float32)
        nan_mask = np.isnan(values)
        if nan_mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                col_medians = np.nanmedian(values, axis=0)
            col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
            values = np.where(nan_mask, col_medians[np.newaxis, :], values)

        # Random jitter: shift context start by 0…jitter timesteps
        jitter_offset = int(self.rng.integers(0, self.jitter + 1))
        v = values[jitter_offset:]  # [context_len + long_len, D]

        ctx_end   = self.context_len
        short_end = ctx_end + self.short_len
        mid_end   = ctx_end + self.mid_len
        long_end  = ctx_end + self.long_len

        return {
            "context":      torch.from_numpy(v[:ctx_end]),
            "target_short": torch.from_numpy(v[ctx_end:short_end]),
            "target_mid":   torch.from_numpy(v[ctx_end:mid_end]),
            "target_long":  torch.from_numpy(v[ctx_end:long_end]),
        }
