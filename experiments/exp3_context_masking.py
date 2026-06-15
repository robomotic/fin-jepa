"""
Experiment 3: Context-Masking Verification

Zero-out specific input channels and measure how well the frozen encoder's
latent representation still captures the economic regime.

Comparison:
  z_full    — unmasked context latent (all channels visible)
  z_masked  — latent from partially-masked context

Metrics:
  1. Cosine similarity(z_masked, z_full): does masking preserve structure?
  2. IC of linear probe (from Exp 1) applied to z_masked: does predictive
     power survive the mask?

Masking ablation table (from plan):
  A. All equity/indices → 0, keep macro
  B. All equity/indices + DXY → 0, keep yields only
  C. All equity/indices → 0, keep GPR only
  D. All equity/indices → 0, keep labor data only
  E. All macro → 0, keep equity only  ← falsifiability check

MLP baseline: a small MLP trained only on macro inputs (never sees equity
prices during training) runs the same masked evaluation. If it matches JEPA,
JEPA is not adding anything beyond direct correlation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.neural_network import MLPRegressor
from torch.utils.data import DataLoader

from data.dataset import FinancialJEPADataset
from experiments.exp1_linear_probe import (
    _collate_fn,
    compute_forward_returns,
    information_coefficient,
    run_linear_probe,
)
from model.jepa import JEPA


# ─── Masked Latent Extraction ─────────────────────────────────────────────────

@torch.no_grad()
def extract_masked_latents(
    encoder: nn.Module,
    dataset: FinancialJEPADataset,
    col_mask: torch.Tensor,   # [D] bool — True = keep, False = zero-out
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract latents with specified columns zeroed out.

    Returns: [N_windows, d_model]
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=_collate_fn)
    encoder.eval().to(device)
    col_mask_f = col_mask.float().to(device)  # [D]

    all_latents = []
    for batch in loader:
        ctx = batch["context"].to(device)               # [B, T, D]
        ctx_masked = ctx * col_mask_f.unsqueeze(0).unsqueeze(0)  # broadcast over B, T
        z = encoder(ctx_masked)                          # [B, N_patches, d_model]
        z_pooled = z.mean(dim=1).cpu().numpy()           # [B, d_model]
        all_latents.append(z_pooled)

    return np.vstack(all_latents)


# ─── Masking Scenarios ────────────────────────────────────────────────────────

def build_masking_scenarios(dataset: FinancialJEPADataset, config: dict) -> dict[str, torch.Tensor]:
    """Return named masking scenarios.

    Each value is a [D] bool tensor: True = channel visible in context.
    """
    series_cfg = config.get("series", {})
    cols = dataset.columns
    D = len(cols)

    def col_selector(**criteria) -> torch.Tensor:
        mask = torch.zeros(D, dtype=torch.bool)
        for i, col in enumerate(cols):
            cfg = series_cfg.get(col, {})
            match = all(cfg.get(k) == v for k, v in criteria.items())
            if match:
                mask[i] = True
        return mask

    equity_mask = col_selector(source="yahoo")  # True = equity/ETF channel

    scenarios = {
        "full":            torch.ones(D, dtype=torch.bool),
        "macro_only":      ~equity_mask,                    # A: zero equity
        "yields_only":     col_selector(pillar=1),          # B: Pillar 1 only
        "gpr_only":        col_selector(pillar=4),          # C: Pillar 4 only
        "labor_only":      col_selector(pillar=6),          # D: Pillar 6 only
        "equity_only":     equity_mask,                     # E: falsifiability
    }
    return scenarios


# ─── MLP Baseline ─────────────────────────────────────────────────────────────

def train_mlp_baseline(
    train_panel: pd.DataFrame,
    train_labels: np.ndarray,
    train_dates: pd.DatetimeIndex,
    macro_only_cols: list[str],
) -> MLPRegressor:
    """Fit a small MLP only on macro (non-equity) inputs."""
    X_raw = train_panel[macro_only_cols].reindex(train_dates)
    # Z-scored panel: NaN from burn-in or publication lag → treat as 0 (mean)
    X = np.nan_to_num(X_raw.values, nan=0.0)
    y = train_labels

    finite = np.isfinite(y)
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
    )
    mlp.fit(X[finite], y[finite])
    return mlp


# ─── Main Experiment ──────────────────────────────────────────────────────────

def run_experiment_3(
    jepa: JEPA,
    train_dataset: FinancialJEPADataset,
    test_dataset: FinancialJEPADataset,
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    prices_test: pd.DataFrame,
    config: dict,
    device: torch.device,
    horizon_days: int = 20,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the full context-masking experiment.

    Returns DataFrame with columns:
      scenario | IC | cosine_sim_to_full
    """
    scenarios = build_masking_scenarios(train_dataset, config)
    encoder = jepa.encoder

    # Full latents for cosine similarity baseline
    full_mask = scenarios["full"]
    latents_train_full = extract_masked_latents(encoder, train_dataset, full_mask, device)
    latents_test_full  = extract_masked_latents(encoder, test_dataset,  full_mask, device)

    # Labels
    exp_cfg = config.get("experiments", {}).get("linear_probe", {})
    target_pairs = exp_cfg.get("target_pairs", [["XLK", "XLF"]])
    num, den = target_pairs[0]
    fwd_returns = compute_forward_returns(prices_test, num, den, horizon_days)

    import pandas as pd
    test_dates  = pd.to_datetime([s["meta"].context_end_date for s in
                                   [test_dataset[i] for i in range(len(test_dataset))]])
    train_dates = pd.to_datetime([s["meta"].context_end_date for s in
                                   [train_dataset[i] for i in range(len(train_dataset))]])

    test_labels  = fwd_returns.reindex(test_dates).values
    train_labels = fwd_returns.reindex(train_dates).values

    results = []

    for scenario_name, mask in scenarios.items():
        logger.info(f"Experiment 3: scenario '{scenario_name}'")

        lat_train = extract_masked_latents(encoder, train_dataset, mask, device)
        lat_test  = extract_masked_latents(encoder, test_dataset,  mask, device)

        # IC via linear probe
        ic = run_linear_probe(lat_train, train_labels, lat_test, test_labels)

        # Cosine similarity to full representation (per-window, then averaged)
        cos_sims = [
            float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            for a, b in zip(lat_test, latents_test_full)
        ]
        mean_cos = float(np.mean(cos_sims))

        results.append({
            "scenario": scenario_name,
            "IC": ic,
            "cosine_sim_to_full": mean_cos,
        })
        logger.info(f"  IC={ic:.4f}  cosine_to_full={mean_cos:.4f}")

    # MLP baseline (macro only)
    series_cfg = config.get("series", {})
    macro_cols = [c for c in train_panel.columns
                  if series_cfg.get(c, {}).get("source") != "yahoo"]

    try:
        mlp = train_mlp_baseline(train_panel, train_labels, train_dates, macro_cols)
        X_test = np.nan_to_num(
            test_panel[macro_cols].reindex(test_dates).values, nan=0.0
        )
        preds = mlp.predict(X_test)
        mlp_ic = information_coefficient(test_labels, preds)
        results.append({
            "scenario": "MLP_baseline_macro_only",
            "IC": mlp_ic,
            "cosine_sim_to_full": float("nan"),
        })
        logger.info(f"  MLP baseline IC={mlp_ic:.4f}")
    except Exception as e:
        logger.warning(f"MLP baseline failed: {e}")

    df = pd.DataFrame(results)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "exp3_masking.csv", index=False)
        logger.info(f"Exp 3 results saved to {output_dir}")

    return df
