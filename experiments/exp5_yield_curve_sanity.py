"""
Experiment 5: Yield Curve Sanity Check

The minimal falsifiability test recommended by the professor:

  Context encoder sees the 2-Year Treasury yield (US02Y) but has the
  10-Year Treasury yield (US10Y) zeroed out.  The predictor must recover
  a target latent that was produced by the target encoder on the FULL
  panel (both yields visible).

  Rationale: GS2 and GS10 are structurally co-integrated — driven by the
  same policy expectations and risk-premium factors.  A healthy encoder
  should map this shared information into a latent space where the
  2Y alone is sufficient to predict the 10Y direction.

Pass criterion:  mean cosine similarity > 0.30
  (conservative floor; chance baseline ≈ 0)

  If the trained model fails this test while a random model also fails,
  the issue is architecture/collapse.
  If the trained model fails but random noise is even lower, the data path
  is likely broken (NaN flooding, wrong column ordering, etc.).

Output: results/exp5_yield_curve_sanity.json
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from data.dataset import FinancialJEPADataset
from model.jepa import JEPA


@torch.no_grad()
def _cosine_sim_batch(z_pred: torch.Tensor, z_target: torch.Tensor) -> np.ndarray:
    """Mean-pool patches then compute cosine similarity per sample.

    z_pred, z_target: [B, N, d_model]
    Returns: [B] cosine similarities.
    """
    p = z_pred.mean(dim=1)    # [B, d]
    t = z_target.mean(dim=1)  # [B, d]
    p_norm = p / (p.norm(dim=1, keepdim=True) + 1e-8)
    t_norm = t / (t.norm(dim=1, keepdim=True) + 1e-8)
    return (p_norm * t_norm).sum(dim=1).cpu().numpy()


def run_experiment_5(
    jepa: JEPA,
    test_panel: pd.DataFrame,
    config: dict,
    device: torch.device,
    output_dir: Optional[Path] = None,
    batch_size: int = 32,
) -> dict:
    """Run the yield curve sanity check.

    Masks US10Y from the context, measures whether the predictor can
    still produce latents aligned with the full-information target encoder.

    Returns a dict with cosine similarities, pass/fail, and baseline comparison.
    """
    model_cfg = config.get("model", {})
    patch_len        = model_cfg.get("patch_len", 21)
    n_patches_ctx    = model_cfg.get("n_patches_context", 9)
    n_patches_tgt    = model_cfg.get("n_patches_target", 3)

    cols = list(test_panel.columns)
    D = len(cols)

    if "US10Y" not in cols:
        logger.warning("US10Y not found in test panel columns — skipping Exp 5")
        return {"skipped": True, "reason": "US10Y column missing"}
    if "US02Y" not in cols:
        logger.warning("US02Y not found in test panel columns — skipping Exp 5")
        return {"skipped": True, "reason": "US02Y column missing"}

    us10y_idx = cols.index("US10Y")
    logger.info(f"Exp 5: masking US10Y (column {us10y_idx}/{D}) from context; "
                f"US02Y remains visible.")

    # Build dataset over the test panel (no masking — we apply it manually below)
    ds = FinancialJEPADataset(
        panel=test_panel,
        config=config,
        patch_len=patch_len,
        n_patches_context=n_patches_ctx,
        n_patches_target=n_patches_tgt,
        stride=5,
        masking_strategy="none",
    )

    if len(ds) == 0:
        logger.warning("Exp 5: test dataset is empty — skipping")
        return {"skipped": True, "reason": "empty test dataset"}

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, drop_last=False,
        collate_fn=lambda samples: {
            "context": torch.stack([s["context"] for s in samples]),
            "target":  torch.stack([s["target"]  for s in samples]),
        },
    )

    jepa.eval()

    # Random-encoder baseline: same architecture, fresh random weights
    random_jepa = copy.deepcopy(jepa)
    for m in random_jepa.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    random_jepa.eval().to(device)
    jepa.to(device)

    trained_cos_sims: list[float] = []
    random_cos_sims:  list[float] = []

    for batch in loader:
        ctx = batch["context"].to(device)   # [B, T_ctx, D]
        tgt = batch["target"].to(device)    # [B, T_tgt, D]

        # Zero out US10Y in context only (target sees full panel)
        ctx_masked = ctx.clone()
        ctx_masked[:, :, us10y_idx] = 0.0

        # Trained model
        z_pred, z_target = jepa(ctx_masked, tgt)
        trained_cos_sims.extend(_cosine_sim_batch(z_pred, z_target).tolist())

        # Random baseline
        z_pred_r, z_target_r = random_jepa(ctx_masked, tgt)
        random_cos_sims.extend(_cosine_sim_batch(z_pred_r, z_target_r).tolist())

    trained_mean = float(np.mean(trained_cos_sims))
    trained_std  = float(np.std(trained_cos_sims))
    random_mean  = float(np.mean(random_cos_sims))
    random_std   = float(np.std(random_cos_sims))
    pass_threshold = 0.30
    passed = trained_mean > pass_threshold

    logger.info(
        f"\nExp 5 — Yield Curve Sanity Check\n"
        f"  Trained  cosine sim: {trained_mean:.4f} ± {trained_std:.4f}\n"
        f"  Random   cosine sim: {random_mean:.4f} ± {random_std:.4f}\n"
        f"  Pass threshold: {pass_threshold}\n"
        f"  Result: {'✓ PASS' if passed else '✗ FAIL'}"
    )

    if not passed:
        if trained_mean > random_mean:
            logger.warning(
                "Trained model beats random but below threshold — "
                "partial learning; check VICReg variance term weights or increase training epochs."
            )
        else:
            logger.warning(
                "Trained model at or below random baseline — "
                "likely representation collapse; verify EMA target encoder updates and "
                "that variance loss (lambda_var) is non-zero."
            )

    result = {
        "trained_cosine_mean": trained_mean,
        "trained_cosine_std":  trained_std,
        "random_cosine_mean":  random_mean,
        "random_cosine_std":   random_std,
        "pass_threshold":      pass_threshold,
        "passed":              passed,
        "n_windows":           len(trained_cos_sims),
        "masked_column":       "US10Y",
        "visible_anchor":      "US02Y",
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "exp5_yield_curve_sanity.json", "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Exp 5 results saved to {output_dir / 'exp5_yield_curve_sanity.json'}")

    return result
