"""
Training entry point for the Financial JEPA world model.

Usage:
    python train.py [--config config/variables.yaml] [--epochs 100]
                    [--batch-size 64] [--lr 3e-4] [--device cuda]
                    [--checkpoint-dir checkpoints/] [--resume]

After training, run experiments via:
    python -m experiments.exp1_linear_probe  (once train.py completes)
    python -m experiments.exp2_latent_arithmetic
    ...

Or run the full evaluation suite:
    python train.py --eval-only --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader

from data.dataset import FinancialJEPADataset
from data.pipeline import build_pipeline, load_config
from model.jepa import JEPA, JEPAConfig


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Financial JEPA")
    p.add_argument("--config",          default="config/variables.yaml")
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--batch-size",      type=int,   default=64)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint-dir",  default="checkpoints")
    p.add_argument("--resume",          action="store_true")
    p.add_argument("--eval-only",       action="store_true")
    p.add_argument("--checkpoint",      default=None,
                   help="Path to checkpoint for --eval-only or --resume")
    p.add_argument("--force-rebuild",   action="store_true",
                   help="Re-download all data even if cache exists")
    p.add_argument("--no-diagnostics",  action="store_true",
                   help="Skip statistical diagnostics (faster first run)")
    p.add_argument("--probe-every",     type=int,   default=5,
                   help="Run IC probe on val set every N epochs (0=disable)")
    p.add_argument("--probe-pair",      default="SPY/HYG",
                   help="Numerator/denominator pair for checkpointing IC (e.g. SPY/HYG)")
    p.add_argument("--probe-horizon",   type=int,   default=20,
                   help="Forward-return horizon in days for checkpointing IC")
    return p.parse_args()


# ─── Dataset Builder ──────────────────────────────────────────────────────────

def build_datasets(splits: dict, config: dict) -> tuple[FinancialJEPADataset, ...]:
    model_cfg = config.get("model", {})
    patch_len      = model_cfg.get("patch_len", 21)
    n_ctx_patches  = model_cfg.get("n_patches_context", 9)
    n_tgt_patches  = model_cfg.get("n_patches_target", 3)

    common = dict(
        config=config,
        patch_len=patch_len,
        n_patches_context=n_ctx_patches,
        n_patches_target=n_tgt_patches,
        stride=5,
    )

    train_ds = FinancialJEPADataset(splits["train"], **common, masking_strategy="none")
    val_ds   = FinancialJEPADataset(splits["val"],   **common, masking_strategy="none")
    test_ds  = FinancialJEPADataset(splits["test"],  **common, masking_strategy="none")

    logger.info(f"Dataset sizes: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_ds, val_ds, test_ds


# ─── Training ─────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = torch.device(args.device)

    # Data
    splits = build_pipeline(
        config_path=args.config,
        force_rebuild=args.force_rebuild,
        run_diagnostics=not args.no_diagnostics,
    )
    train_ds, val_ds, test_ds = build_datasets(splits, config)

    # Full panel for forward-return labels used by the online IC probe
    prices_full = pd.concat(
        [splits["train"], splits["val"], splits["test"]]
    ).sort_index()
    prices_full = prices_full[~prices_full.index.duplicated(keep="last")]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
        collate_fn=_collate_fn,
    )

    # Model
    n_features = len(train_ds.columns)
    model_cfg = config.get("model", {})
    jepa_cfg = JEPAConfig(
        n_features=n_features,
        patch_len=model_cfg.get("patch_len", 21),
        n_patches_context=model_cfg.get("n_patches_context", 9),
        n_patches_target=model_cfg.get("n_patches_target", 3),
        d_model=model_cfg.get("d_model", 256),
        n_heads=model_cfg.get("n_heads", 8),
        n_encoder_layers=model_cfg.get("n_layers", 6),
        d_ff=model_cfg.get("d_ff", 1024),
        dropout=model_cfg.get("dropout", 0.1),
        tau_start=model_cfg.get("tau_start", 0.996),
        tau_end=model_cfg.get("tau_end", 0.9999),
    )

    jepa = JEPA(jepa_cfg).to(device)
    logger.info(f"JEPA parameters: {sum(p.numel() for p in jepa.parameters()):,}")

    # Fix tau annealing to span the actual number of optimizer steps
    actual_steps = len(train_loader) * args.epochs
    jepa.target_encoder.total_steps = actual_steps
    logger.info(f"Target encoder tau annealing over {actual_steps} steps ({jepa_cfg.tau_start:.4f}→{jepa_cfg.tau_end:.4f})")

    # Optimiser — do not include target encoder (no grad)
    trainable = list(jepa.encoder.parameters()) + list(jepa.predictor.parameters())
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = max(1, args.epochs // 10)
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val_loss = float("inf")
    best_val_ic   = float("-inf")

    # Parse probe pair
    _probe_parts = args.probe_pair.split("/")
    probe_num, probe_den = _probe_parts[0].strip(), _probe_parts[1].strip()
    use_ic_checkpoint = args.probe_every > 0

    if args.resume and args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        jepa.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_val_ic   = ckpt.get("best_val_ic",   float("-inf"))
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        jepa.train()
        train_metrics: dict[str, float] = {}

        for batch in train_loader:
            ctx = batch["context"].to(device)
            tgt = batch["target"].to(device)

            optimizer.zero_grad()
            loss, metrics = jepa.training_step(ctx, tgt)

            if not torch.isfinite(loss):
                logger.warning("NaN/Inf loss — skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            jepa.update_target_encoder()

            for k, v in metrics.items():
                train_metrics[k] = train_metrics.get(k, 0.0) + v

        n_batches = max(len(train_loader), 1)
        train_metrics = {k: v / n_batches for k, v in train_metrics.items()}

        # Validation
        val_loss = evaluate(jepa, val_loader, device)
        scheduler.step()

        # Online IC probe (cheap: ~13 forward passes on train + 1 on val)
        val_ic = float("nan")
        run_probe = (
            use_ic_checkpoint
            and ((epoch + 1) % args.probe_every == 0 or epoch == 0)
        )
        if run_probe:
            val_ic = _val_probe_ic(
                jepa, train_ds, val_ds, prices_full, device,
                probe_num, probe_den, args.probe_horizon,
            )

        ic_str = f"  val_ic={val_ic:+.4f}" if not (val_ic != val_ic) else ""
        logger.info(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={train_metrics.get('loss_total', 0):.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"tau={jepa.target_encoder.current_tau:.5f}"
            f"{ic_str}"
        )

        # Checkpoint: prefer IC when available after warmup (skip early noisy IC spikes)
        ic_warmup_epochs = 20
        if use_ic_checkpoint and not math.isnan(val_ic) and (epoch + 1) >= ic_warmup_epochs:
            is_best = val_ic > best_val_ic
            if is_best:
                best_val_ic = val_ic
        else:
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

        ckpt = {
            "epoch": epoch,
            "model": jepa.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_ic": best_val_ic,
            "config": model_cfg,
            "n_features": n_features,
            "columns": train_ds.columns,
        }
        torch.save(ckpt, ckpt_dir / "latest.pt")
        if is_best:
            torch.save(ckpt, ckpt_dir / "best.pt")
            if use_ic_checkpoint and not math.isnan(val_ic):
                logger.info(f"  ↳ New best model (val_ic={val_ic:+.4f})")
            else:
                logger.info(f"  ↳ New best model (val_loss={val_loss:.4f})")

    logger.info("Training complete.")
    # Load best checkpoint before running experiments
    best_ckpt_path = ckpt_dir / "best.pt"
    if best_ckpt_path.exists():
        logger.info(f"Loading best checkpoint for experiments: {best_ckpt_path}")
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        jepa.load_state_dict(best_ckpt["model"])
        logger.info(f"  best epoch={best_ckpt.get('epoch', '?')}, val_ic={best_ckpt.get('best_val_ic', float('nan')):+.4f}")
    _run_all_experiments(jepa, splits, train_ds, val_ds, test_ds, config, device)


@torch.no_grad()
def _val_probe_ic(
    jepa: JEPA,
    train_ds: FinancialJEPADataset,
    val_ds: FinancialJEPADataset,
    prices_full: pd.DataFrame,
    device: torch.device,
    num: str,
    den: str,
    horizon: int,
) -> float:
    """Fit Ridge on frozen train latents; return Spearman IC on val latents."""
    from experiments.exp1_linear_probe import (
        extract_latents, compute_forward_returns, run_linear_probe,
    )
    if num not in prices_full.columns or den not in prices_full.columns:
        return float("nan")
    latents_tr, dates_tr = extract_latents(jepa.encoder, train_ds, device)
    latents_val, dates_val = extract_latents(jepa.encoder, val_ds, device)
    fwd = compute_forward_returns(prices_full, num, den, horizon)
    labels_tr  = fwd.reindex(pd.to_datetime(dates_tr)).values
    labels_val = fwd.reindex(pd.to_datetime(dates_val)).values
    return run_linear_probe(latents_tr, labels_tr, latents_val, labels_val)


@torch.no_grad()
def evaluate(jepa: JEPA, loader: DataLoader, device: torch.device) -> float:
    jepa.eval()
    total_loss, n = 0.0, 0
    from model.jepa import vicreg_loss
    for batch in loader:
        ctx = batch["context"].to(device)
        tgt = batch["target"].to(device)
        z_pred, z_target = jepa(ctx, tgt)
        loss, _ = vicreg_loss(z_pred, z_target.detach(),
                               lambda_inv=jepa.cfg.lambda_inv,
                               lambda_var=jepa.cfg.lambda_var,
                               lambda_cov=jepa.cfg.lambda_cov)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def _collate_fn(samples):
    context = torch.stack([s["context"] for s in samples])
    target  = torch.stack([s["target"]  for s in samples])
    mask    = torch.stack([s["mask"]    for s in samples])
    return {"context": context, "target": target, "mask": mask,
            "meta": [s["meta"] for s in samples]}


# ─── Post-Training Experiments ────────────────────────────────────────────────

def _run_all_experiments(
    jepa: JEPA,
    splits: dict,
    train_ds: FinancialJEPADataset,
    val_ds: FinancialJEPADataset,
    test_ds: FinancialJEPADataset,
    config: dict,
    device: torch.device,
) -> None:
    from experiments.exp1_linear_probe import run_experiment_1, print_summary
    from experiments.exp2_latent_arithmetic import run_experiment_2
    from experiments.exp3_context_masking import run_experiment_3
    from experiments.exp4_geopolitical_transfer import run_experiment_4

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Freeze encoder for all experiments
    for p in jepa.encoder.parameters():
        p.requires_grad_(False)
    jepa.eval()

    # Full z-scored panel spanning train+val+test for forward return computation
    prices_full = pd.concat([splits["train"], splits["val"], splits["test"]]).sort_index()
    prices_full = prices_full[~prices_full.index.duplicated(keep="last")]

    # Exp 1
    logger.info("\n══ Running Experiment 1: Linear Probing ══")
    exp1_results = run_experiment_1(
        jepa, train_ds, test_ds, prices_full, config, device,
        output_path=results_dir / "exp1_ic_results.csv",
    )
    print_summary(exp1_results)

    # Exp 2
    logger.info("\n══ Running Experiment 2: Latent Arithmetic ══")
    exp2_result = run_experiment_2(
        jepa, train_ds, splits["train"], config, device,
        output_dir=results_dir / "exp2",
    )

    # Exp 3
    logger.info("\n══ Running Experiment 3: Context Masking ══")
    exp3_results = run_experiment_3(
        jepa, train_ds, test_ds,
        splits["train"], splits["test"], prices_full,
        config, device, horizon_days=20,
        output_dir=results_dir / "exp3",
    )
    logger.info(f"\nExp 3 summary:\n{exp3_results.to_string(index=False)}")

    # Exp 4 — needs val+test panel so the event window ending 2022-02-24
    # has enough history (189 context days; test alone starts 2022-01-24)
    val_test_panel = pd.concat([splits["val"], splits["test"]]).sort_index()
    val_test_panel = val_test_panel[~val_test_panel.index.duplicated(keep="last")]

    logger.info("\n══ Running Experiment 4: Geopolitical Transfer ══")
    v_gpr = exp2_result.get("v_gpr")
    exp4_result = run_experiment_4(
        jepa, val_test_panel, config, device,
        v_gpr_shock=v_gpr,
        output_dir=results_dir / "exp4",
    )

    logger.info(f"\nExp 4 — Δz norm: {exp4_result['delta_z_norm']:.4f}")
    if exp4_result["cosine_to_shock"] is not None:
        logger.info(f"Exp 4 — cos(Δz, v_shock): {exp4_result['cosine_to_shock']:.4f}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint <path>")
        config = load_config(args.config)
        device = torch.device(args.device)
        ckpt   = torch.load(args.checkpoint, map_location=device)

        jepa_cfg = JEPAConfig(
            n_features=ckpt["n_features"],
            **{k: v for k, v in ckpt["config"].items()
               if k in JEPAConfig.__dataclass_fields__},
        )
        jepa = JEPA(jepa_cfg).to(device)
        jepa.load_state_dict(ckpt["model"])

        splits   = build_pipeline(config_path=args.config, force_rebuild=False)
        train_ds, val_ds, test_ds = build_datasets(splits, config)
        _run_all_experiments(jepa, splits, train_ds, val_ds, test_ds, config, device)
    else:
        train(args)
