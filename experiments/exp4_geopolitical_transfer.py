"""
Experiment 4: Geopolitical Regime Transfer Test

The hardest test: JEPA trained on 2000–2019 must generalise to an
out-of-sample geopolitical shock whose magnitude was never seen in training.

Event date: 2022-02-24 (Russia invades Ukraine).
GPR_DAILY on this date was near historical record — far above training range.

Protocol:
  1. Load JEPA trained on 2000-2019 (no 2022 data seen).
  2. Build a context window ending on 2022-02-24, masking all equity channels
     (only GPR, DXY, TIPS breakevens visible).
  3. Obtain z_event = encoder(masked_context)   — mean-pooled latent
  4. Obtain z̄_baseline = mean latent over Jan 2022 windows (same mask)
  5. Δz = z_event − z̄_baseline
  6. Measure:
       (a) cosine(Δz, v_GPR_shock) — alignment with Exp 2 shock vector
       (b) Linear probe scores for ITA and EEM (from Exp 1 probes)
           applied to z_event vs z̄_baseline

Falsifiability comparison: repeat with a supervised LSTM trained on
next-day returns. The LSTM likely saturates at smaller shocks since
it learned a magnitude-fitted mapping, not a directional vector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from data.dataset import FinancialJEPADataset
from experiments.exp3_context_masking import build_masking_scenarios, extract_masked_latents
from model.jepa import JEPA


def _mean_pool(latents_3d: torch.Tensor) -> np.ndarray:
    """[B, N, d] → [B, d]"""
    return latents_3d.mean(dim=1).cpu().numpy()


@torch.no_grad()
def encode_single_window(
    encoder: nn.Module,
    panel: pd.DataFrame,
    context_end_date: str,
    context_len: int,
    col_mask: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Encode a single context window ending at context_end_date.

    Returns [d_model] latent vector.
    """
    end = pd.Timestamp(context_end_date)
    dates = panel.index[panel.index <= end]

    if len(dates) < context_len:
        raise ValueError(
            f"Not enough history before {context_end_date} "
            f"(need {context_len} rows, have {len(dates)})"
        )

    window = panel.iloc[panel.index.get_loc(dates[-context_len]) :
                        panel.index.get_loc(dates[-1]) + 1]

    if len(window) < context_len:
        raise ValueError(f"Window too short: {len(window)}")

    values = window.values.astype(np.float32)
    values = np.where(np.isnan(values), 0.0, values)

    x = torch.from_numpy(values).unsqueeze(0).to(device)   # [1, T, D]
    mask_f = col_mask.float().to(device)
    x = x * mask_f.unsqueeze(0).unsqueeze(0)

    encoder.eval()
    z = encoder(x)                   # [1, N_patches, d]
    return z.mean(dim=1).squeeze(0).cpu().numpy()   # [d]


def run_experiment_4(
    jepa: JEPA,
    test_panel: pd.DataFrame,          # normalised panel (includes 2022)
    config: dict,
    device: torch.device,
    v_gpr_shock: Optional[np.ndarray] = None,   # from Exp 2; if None, skip (a)
    linear_probes: Optional[dict] = None,         # from Exp 1; keys: ticker names
    output_dir: Optional[Path] = None,
) -> dict:
    """Run the geopolitical transfer test.

    Returns dict with all computed metrics.
    """
    event_date    = config["splits"]["geopolitical_event_date"]  # "2022-02-24"
    model_cfg     = config.get("model", {})
    context_len   = model_cfg.get("patch_len", 21) * model_cfg.get("n_patches_context", 9)
    baseline_days = config.get("experiments", {}).get(
        "geopolitical_transfer", {}
    ).get("baseline_window_days", 21)

    encoder = jepa.encoder

    # Build the GPR-only mask (keep GPR + DXY + TIPS, zero all equity)
    series_cfg = config.get("series", {})
    cols = test_panel.columns.tolist()
    D = len(cols)

    macro_geopolitical_mask = torch.zeros(D, dtype=torch.bool)
    # Only GPR-source series (not equity tickers in pillar 4 like ITA)
    # plus explicit macro rate/FX inputs
    keep_series = {"TIPS5Y", "TIPS5Y5Y", "DXY"}
    for i, col in enumerate(cols):
        cfg = series_cfg.get(col, {})
        if cfg.get("source") == "gpr" or col in keep_series:
            macro_geopolitical_mask[i] = True

    logger.info(
        f"Geopolitical mask: {macro_geopolitical_mask.sum().item()} / {D} channels visible"
    )
    logger.info(f"Visible: {[c for c, m in zip(cols, macro_geopolitical_mask) if m]}")

    # Encode the event window
    logger.info(f"Encoding event window ending {event_date}")
    z_event = encode_single_window(
        encoder, test_panel, event_date, context_len, macro_geopolitical_mask, device
    )

    # Encode baseline windows (Jan 2022)
    baseline_end = pd.Timestamp(event_date) - pd.offsets.BDay(1)
    baseline_start = baseline_end - pd.offsets.BDay(baseline_days - 1)
    baseline_dates = test_panel.index[
        (test_panel.index >= baseline_start) & (test_panel.index <= baseline_end)
    ]

    baseline_latents = []
    for d in baseline_dates:
        try:
            z = encode_single_window(
                encoder, test_panel, str(d.date()), context_len,
                macro_geopolitical_mask, device
            )
            baseline_latents.append(z)
        except Exception:
            pass

    if not baseline_latents:
        raise ValueError("No valid baseline windows in Jan 2022. Check test_panel coverage.")

    z_baseline_mean = np.mean(baseline_latents, axis=0)   # [d]
    z_baseline_std  = np.std(baseline_latents, axis=0)

    delta_z = z_event - z_baseline_mean

    # (a) Cosine similarity to GPR shock vector from Exp 2
    cos_to_shock = None
    if v_gpr_shock is not None:
        cos_to_shock = float(
            np.dot(delta_z, v_gpr_shock) /
            (np.linalg.norm(delta_z) * np.linalg.norm(v_gpr_shock) + 1e-8)
        )
        logger.info(f"Δz cosine to v_GPR_shock: {cos_to_shock:.4f}")
        if cos_to_shock >= 0.5:
            logger.info("✓ Strong alignment with geopolitical shock vector")
        else:
            logger.warning("✗ Weak alignment — model may not have generalised GPR structure")

    # (b) Linear probe scores
    probe_scores = {}
    if linear_probes:
        for ticker, probe in linear_probes.items():
            score_event    = float(probe.predict(z_event.reshape(1, -1))[0])
            score_baseline = float(probe.predict(z_baseline_mean.reshape(1, -1))[0])
            z_score = (score_event - score_baseline) / (np.std([
                probe.predict(z.reshape(1, -1))[0] for z in baseline_latents
            ]) + 1e-8)
            probe_scores[ticker] = {
                "event_score":    score_event,
                "baseline_score": score_baseline,
                "z_score":        z_score,
            }
            direction = "↑" if z_score > 0 else "↓"
            logger.info(f"  Probe [{ticker}]: event={score_event:.4f}  "
                        f"baseline={score_baseline:.4f}  z={z_score:.2f} {direction}")

    result = {
        "event_date":       event_date,
        "z_event":          z_event,
        "z_baseline_mean":  z_baseline_mean,
        "z_baseline_std":   z_baseline_std,
        "delta_z":          delta_z,
        "delta_z_norm":     float(np.linalg.norm(delta_z)),
        "cosine_to_shock":  cos_to_shock,
        "probe_scores":     probe_scores,
        "n_baseline_windows": len(baseline_latents),
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "exp4_z_event.npy",    z_event)
        np.save(output_dir / "exp4_delta_z.npy",    delta_z)
        np.save(output_dir / "exp4_z_baseline.npy", z_baseline_mean)

        summary = {
            "event_date":       event_date,
            "delta_z_norm":     result["delta_z_norm"],
            "cosine_to_shock":  cos_to_shock,
            "probe_scores":     {k: {kk: float(vv) for kk, vv in v.items()}
                                 for k, v in probe_scores.items()},
            "n_baseline_windows": len(baseline_latents),
        }
        import json
        with open(output_dir / "exp4_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Exp 4 outputs saved to {output_dir}")

    return result
