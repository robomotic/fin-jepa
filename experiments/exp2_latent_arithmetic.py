"""
Experiment 2: Latent Vector Arithmetic (The Macro Shock Test)

Uses GPR_DAILY to define shock / calm periods objectively (pre-registered criterion):
  shock: GPR_GLOBAL > 90th percentile of training distribution
  calm:  GPR_GLOBAL < 25th percentile of training distribution

Computes:
  z̄_shock = mean latent embedding over shock windows
  z̄_calm  = mean latent embedding over calm windows
  v_shock = z̄_shock − z̄_calm       ← the "geopolitical risk vector"

Separately for GPRA (Acts) and GPRT (Threats):
  v_GPRA, v_GPRT — should point in the same direction, GPRA with larger norm

Perturbation test:
  Repeat with ±5-day window shifts and ±10-percentile thresholds.
  If direction reverses, the effect is not robust.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from data.dataset import FinancialJEPADataset
from experiments.exp1_linear_probe import extract_latents, _collate_fn
from model.jepa import JEPA


def _get_threshold(series: pd.Series, pct: float) -> float:
    return float(np.nanpercentile(series.values, pct))


def compute_shock_vector(
    jepa: JEPA,
    train_dataset: FinancialJEPADataset,
    train_panel: pd.DataFrame,          # normalised training panel
    config: dict,
    device: torch.device,
) -> dict:
    """Compute geopolitical shock vector from training data.

    Returns dict with:
      v_gpr:         [d_model] — overall shock vector
      v_gpra:        [d_model] — Acts shock vector
      v_gprt:        [d_model] — Threats shock vector
      z_bar_shock:   [d_model] — mean embedding of shock windows
      z_bar_calm:    [d_model] — mean embedding of calm windows
      thresholds:    {shock_pct, calm_pct, shock_val, calm_val}
    """
    exp_cfg = config.get("experiments", {}).get("latent_arithmetic", {})
    shock_pct = exp_cfg.get("gpr_shock_pct", 90)
    calm_pct  = exp_cfg.get("gpr_calm_pct", 25)

    if "GPR_GLOBAL" not in train_panel.columns:
        raise ValueError("GPR_GLOBAL not in training panel. Check data pipeline.")

    gpr_series = train_panel["GPR_GLOBAL"].dropna()
    shock_val  = _get_threshold(gpr_series, shock_pct)
    calm_val   = _get_threshold(gpr_series, calm_pct)

    logger.info(f"GPR shock threshold (p{shock_pct}): {shock_val:.2f}")
    logger.info(f"GPR calm  threshold (p{calm_pct}):  {calm_val:.2f}")

    # Extract all training latents
    jepa.encoder.eval()
    latents, dates = extract_latents(jepa.encoder, train_dataset, device)
    dates_ts = pd.to_datetime(dates)

    # Get GPR value at each window's context_end_date
    gpr_at_dates = train_panel["GPR_GLOBAL"].reindex(dates_ts)

    shock_mask = gpr_at_dates.values > shock_val
    calm_mask  = gpr_at_dates.values < calm_val

    logger.info(f"Shock windows: {shock_mask.sum()} / {len(shock_mask)}")
    logger.info(f"Calm  windows: {calm_mask.sum()} / {len(calm_mask)}")

    if shock_mask.sum() < 5 or calm_mask.sum() < 5:
        raise ValueError("Too few shock/calm windows. Adjust thresholds.")

    z_bar_shock = latents[shock_mask].mean(axis=0)  # [d_model]
    z_bar_calm  = latents[calm_mask].mean(axis=0)

    v_gpr = z_bar_shock - z_bar_calm

    # Repeat for GPRA and GPRT if available
    v_gpra = v_gprt = None
    for sub_col, attr_name in [("GPRA", "v_gpra"), ("GPRT", "v_gprt")]:
        if sub_col not in train_panel.columns:
            continue
        sub_series = train_panel[sub_col].reindex(dates_ts)
        sub_shock_val = _get_threshold(sub_series.dropna(), shock_pct)
        sub_calm_val  = _get_threshold(sub_series.dropna(), calm_pct)
        sub_shock = sub_series.values > sub_shock_val
        sub_calm  = sub_series.values < sub_calm_val
        if sub_shock.sum() >= 5 and sub_calm.sum() >= 5:
            if attr_name == "v_gpra":
                v_gpra = latents[sub_shock].mean(0) - latents[sub_calm].mean(0)
            else:
                v_gprt = latents[sub_shock].mean(0) - latents[sub_calm].mean(0)

    result = {
        "v_gpr": v_gpr,
        "v_gpra": v_gpra,
        "v_gprt": v_gprt,
        "z_bar_shock": z_bar_shock,
        "z_bar_calm": z_bar_calm,
        "thresholds": {
            "shock_pct": shock_pct, "shock_val": float(shock_val),
            "calm_pct":  calm_pct,  "calm_val":  float(calm_val),
        },
    }

    # GPRA vs GPRT direction check
    if v_gpra is not None and v_gprt is not None:
        cos_sim = float(np.dot(v_gpra, v_gprt) /
                        (np.linalg.norm(v_gpra) * np.linalg.norm(v_gprt) + 1e-8))
        norm_a = float(np.linalg.norm(v_gpra))
        norm_t = float(np.linalg.norm(v_gprt))
        logger.info(f"GPRA vs GPRT cosine similarity: {cos_sim:.4f}")
        logger.info(f"||v_GPRA|| = {norm_a:.4f}  ||v_GPRT|| = {norm_t:.4f}")
        result["gpra_gprt_cosine"] = cos_sim
        result["gpra_norm"] = norm_a
        result["gprt_norm"] = norm_t

    return result


def perturbation_test(
    jepa: JEPA,
    train_dataset: FinancialJEPADataset,
    train_panel: pd.DataFrame,
    config: dict,
    device: torch.device,
    base_result: dict,
) -> pd.DataFrame:
    """Repeat shock vector computation with perturbed thresholds.

    Checks if v_gpr direction is stable under ±10-percentile shifts.
    """
    exp_cfg = config.get("experiments", {}).get("latent_arithmetic", {})
    base_shock = exp_cfg.get("gpr_shock_pct", 90)
    base_calm  = exp_cfg.get("gpr_calm_pct", 25)
    perturb    = exp_cfg.get("perturbation_pct", 10)

    v_base = base_result["v_gpr"]
    rows = []

    for d_shock in [-perturb, 0, perturb]:
        for d_calm in [-perturb, 0, perturb]:
            shock_pct = max(base_shock + d_shock, base_calm + 5)
            calm_pct  = max(base_calm  + d_calm,  5)
            if shock_pct <= calm_pct:
                continue

            cfg_copy = {**config, "experiments": {
                **config.get("experiments", {}),
                "latent_arithmetic": {
                    **exp_cfg,
                    "gpr_shock_pct": shock_pct,
                    "gpr_calm_pct": calm_pct,
                },
            }}
            try:
                result = compute_shock_vector(
                    jepa, train_dataset, train_panel, cfg_copy, device
                )
                v = result["v_gpr"]
                cos_sim = float(np.dot(v, v_base) /
                                (np.linalg.norm(v) * np.linalg.norm(v_base) + 1e-8))
                rows.append({
                    "shock_pct": shock_pct, "calm_pct": calm_pct,
                    "cosine_to_base": cos_sim,
                    "robust": cos_sim > 0.5,
                })
            except Exception as e:
                logger.warning(f"Perturbation ({shock_pct},{calm_pct}) failed: {e}")

    df = pd.DataFrame(rows)
    logger.info(f"Perturbation test: {df['robust'].mean():.1%} of perturbations are robust (cos>0.5)")
    return df


def run_experiment_2(
    jepa: JEPA,
    train_dataset: FinancialJEPADataset,
    train_panel: pd.DataFrame,
    config: dict,
    device: torch.device,
    output_dir: Optional[Path] = None,
) -> dict:
    """Full Experiment 2: shock vector + perturbation test."""
    logger.info("Experiment 2: Computing GPR shock vector")
    result = compute_shock_vector(jepa, train_dataset, train_panel, config, device)

    v = result["v_gpr"]
    logger.info(f"Shock vector L2 norm: {np.linalg.norm(v):.4f}")
    logger.info(f"Shock vector top-5 dimensions: {np.argsort(np.abs(v))[::-1][:5].tolist()}")

    logger.info("Experiment 2: Running perturbation test")
    perturb_df = perturbation_test(jepa, train_dataset, train_panel, config, device, result)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        perturb_df.to_csv(output_dir / "exp2_perturbation.csv", index=False)
        np.save(output_dir / "v_gpr.npy", v)
        if result["v_gpra"] is not None:
            np.save(output_dir / "v_gpra.npy", result["v_gpra"])
        if result["v_gprt"] is not None:
            np.save(output_dir / "v_gprt.npy", result["v_gprt"])
        logger.info(f"Exp 2 outputs saved to {output_dir}")

    result["perturbation_df"] = perturb_df
    return result
