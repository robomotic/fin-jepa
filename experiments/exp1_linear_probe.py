"""
Experiment 1: Linear Probing Test

Procedure:
  1. Load a trained JEPA model (encoder weights frozen).
  2. Extract patch latents for each window in the val/test set.
  3. Pool patches → single vector per window.
  4. Train a strictly linear head (no activation) to predict the n-day
     forward return of a target ratio (e.g. XLK/XLF, GLD/EEM).
  5. Evaluate with Information Coefficient (Spearman IC) at multiple horizons.
  6. Repeat for all three baseline encoders and compare.

The IC comparison table tells you whether JEPA's latent space captures
regime information beyond what a random projection or raw features contain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

from data.dataset import FinancialJEPADataset
from experiments.baselines import (
    RawFeaturesEncoder,
    make_random_encoder,
)
from model.jepa import JEPA, JEPAConfig


# ─── Latent Extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_latents(
    encoder: nn.Module,
    dataset: FinancialJEPADataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """Pass all windows through encoder, mean-pool patches.

    Returns:
        latents: [N_windows, d_model]
        dates:   list of context_end_date strings
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=_collate_fn)
    encoder.eval().to(device)

    all_latents, all_dates = [], []
    for batch in loader:
        ctx = batch["context"].to(device)           # [B, T_ctx, D]
        z = encoder(ctx)                             # [B, N_patches, d_model]
        z_pooled = z.mean(dim=1).cpu().numpy()       # [B, d_model]
        all_latents.append(z_pooled)
        all_dates.extend(batch["meta_context_end"])

    return np.vstack(all_latents), all_dates


def _collate_fn(samples):
    """Custom collate that handles SampleMeta dataclasses."""
    context = torch.stack([s["context"] for s in samples])
    target  = torch.stack([s["target"]  for s in samples])
    mask    = torch.stack([s["mask"]    for s in samples])
    meta_context_end = [s["meta"].context_end_date for s in samples]
    return {
        "context": context,
        "target":  target,
        "mask":    mask,
        "meta_context_end": meta_context_end,
    }


# ─── Forward Returns ──────────────────────────────────────────────────────────

def compute_forward_returns(
    prices: pd.DataFrame,
    numerator: str,
    denominator: str,
    horizon_days: int,
) -> pd.Series:
    """Compute h-day forward return of numerator/denominator ratio.

    Works with z-scored log-returns (not raw prices): the h-day forward
    return is approximated as the rolling sum of future daily spread values.
    Monotonically related to the true log-return of the ratio for IC ranking.
    """
    spread = prices[numerator] - prices[denominator]
    return spread.rolling(horizon_days).sum().shift(-horizon_days)


# ─── IC Metric ────────────────────────────────────────────────────────────────

def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predictions and actual returns."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 10:
        return float("nan")
    ic, _ = spearmanr(y_pred[mask], y_true[mask])
    return float(ic)


# ─── Linear Probe ─────────────────────────────────────────────────────────────

def run_linear_probe(
    latents_train: np.ndarray,
    labels_train: np.ndarray,
    latents_test: np.ndarray,
    labels_test: np.ndarray,
    alpha: float = 1.0,  # Ridge regularisation
) -> float:
    """Fit Ridge(alpha) on train, evaluate IC on test."""
    mask_train = np.isfinite(labels_train)
    mask_test  = np.isfinite(labels_test)

    if mask_train.sum() < 30 or mask_test.sum() < 10:
        return float("nan")

    probe = Ridge(alpha=alpha, fit_intercept=True)
    probe.fit(latents_train[mask_train], labels_train[mask_train])
    preds = probe.predict(latents_test[mask_test])
    return information_coefficient(labels_test[mask_test], preds)


# ─── Main Experiment ──────────────────────────────────────────────────────────

def run_experiment_1(
    jepa: JEPA,
    train_dataset: FinancialJEPADataset,
    test_dataset: FinancialJEPADataset,
    prices_test: pd.DataFrame,           # raw prices for forward return calc
    config: dict,
    device: torch.device = torch.device("cpu"),
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the full linear probing experiment.

    Returns a DataFrame with columns:
      encoder | target_pair | horizon_days | IC
    """
    exp_cfg = config.get("experiments", {}).get("linear_probe", {})
    horizons = exp_cfg.get("horizons_days", [1, 5, 20, 60])
    target_pairs = exp_cfg.get("target_pairs", [["XLK", "XLF"]])

    jepa_cfg = jepa.cfg

    # Build all encoders to evaluate
    encoders = {
        "JEPA":          jepa.encoder,
        "Random":        make_random_encoder(jepa_cfg),
        "RawFeatures":   RawFeaturesEncoder(jepa_cfg.n_features, jepa_cfg.patch_len, jepa_cfg.d_model),
    }

    results = []

    for enc_name, encoder in encoders.items():
        logger.info(f"Extracting latents — encoder: {enc_name}")
        encoder.eval()
        latents_train, dates_train = extract_latents(encoder, train_dataset, device)
        latents_test,  dates_test  = extract_latents(encoder, test_dataset,  device)

        dates_train_ts = pd.to_datetime(dates_train)
        dates_test_ts  = pd.to_datetime(dates_test)

        for pair in target_pairs:
            num, den = pair
            if num not in prices_test.columns or den not in prices_test.columns:
                logger.warning(f"Skipping pair {pair}: not in prices_test columns")
                continue

            for horizon in horizons:
                fwd_returns = compute_forward_returns(prices_test, num, den, horizon)

                # Align labels to window dates
                train_labels = fwd_returns.reindex(dates_train_ts).values
                test_labels  = fwd_returns.reindex(dates_test_ts).values

                ic = run_linear_probe(latents_train, train_labels, latents_test, test_labels)

                results.append({
                    "encoder":     enc_name,
                    "target_pair": f"{num}/{den}",
                    "horizon_days": horizon,
                    "IC":          ic,
                })
                logger.info(f"  {enc_name} | {num}/{den} | {horizon}d → IC={ic:.4f}")

    df = pd.DataFrame(results)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Exp 1 results saved to {output_path}")

    return df


def print_summary(results: pd.DataFrame) -> None:
    """Print a pivot table of IC by encoder × horizon."""
    for pair in results["target_pair"].unique():
        sub = results[results["target_pair"] == pair]
        pivot = sub.pivot(index="encoder", columns="horizon_days", values="IC")
        print(f"\n── {pair} ──")
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
