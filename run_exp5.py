"""
Experiment 5 standalone runner — 2-series isolation test.

Trains a fresh JEPA model on ONLY the 2-Year and 10-Year Treasury yields
(US02Y / US10Y), then runs the yield curve sanity check (mask US10Y from
context, predict from US02Y alone).

Using only 2 series removes any possibility that the 43-feature model is
leaking information through other channels.  If JEPA passes this test on
a 2-variable panel, the architecture is genuinely learning co-movement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import FinancialJEPADataset
from data.pipeline import build_pipeline, load_config
from model.jepa import JEPA, JEPAConfig, vicreg_loss
from experiments.exp5_yield_curve_sanity import run_experiment_5

# ── Config ────────────────────────────────────────────────────────────────────
SERIES      = ["US10Y", "US02Y"]   # the only columns the model will see
EPOCHS      = 200
BATCH_SIZE  = 64
LR          = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH   = Path("checkpoints/exp5_2series.pt")

# Smaller architecture — 2 features don't need the full 256-dim encoder
MODEL_OVERRIDES = dict(
    d_model=128,
    n_heads=4,
    n_encoder_layers=4,
    d_ff=512,
    dropout=0.1,
    tau_start=0.996,
    tau_end=0.996,
)


def _collate(samples):
    return {
        "context": torch.stack([s["context"] for s in samples]),
        "target":  torch.stack([s["target"]  for s in samples]),
    }


def subset_panel(panel: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SERIES if c not in panel.columns]
    if missing:
        raise KeyError(f"Series not found in panel: {missing}")
    return panel[SERIES].copy()


def build_loader(panel: pd.DataFrame, config: dict,
                 shuffle: bool, batch_size: int) -> DataLoader:
    model_cfg = config.get("model", {})
    ds = FinancialJEPADataset(
        panel=panel,
        config=config,
        patch_len=model_cfg.get("patch_len", 21),
        n_patches_context=model_cfg.get("n_patches_context", 9),
        n_patches_target=model_cfg.get("n_patches_target", 3),
        stride=5,
        masking_strategy="none",
    )
    logger.info(f"  {'train' if shuffle else 'eval'} dataset: {len(ds)} windows")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=shuffle, collate_fn=_collate)


def build_model(config: dict) -> JEPA:
    model_cfg = config.get("model", {})
    cfg = JEPAConfig(
        n_features=len(SERIES),
        patch_len=model_cfg.get("patch_len", 21),
        n_patches_context=model_cfg.get("n_patches_context", 9),
        n_patches_target=model_cfg.get("n_patches_target", 3),
        **MODEL_OVERRIDES,
    )
    return JEPA(cfg)


@torch.no_grad()
def evaluate(jepa: JEPA, loader: DataLoader) -> float:
    jepa.eval()
    total, n = 0.0, 0
    for batch in loader:
        ctx = batch["context"].to(DEVICE)
        tgt = batch["target"].to(DEVICE)
        zp, zt = jepa(ctx, tgt)
        loss, _ = vicreg_loss(zp, zt.detach())
        total += loss.item(); n += 1
    return total / max(n, 1)


def train(config: dict, splits: dict) -> JEPA:
    logger.info(f"\n{'='*60}")
    logger.info(f"Training 2-series JEPA on {SERIES}")
    logger.info(f"{'='*60}")

    train_panel = subset_panel(splits["train"])
    val_panel   = subset_panel(splits["val"])

    train_loader = build_loader(train_panel, config, shuffle=True,  batch_size=BATCH_SIZE)
    val_loader   = build_loader(val_panel,   config, shuffle=False, batch_size=BATCH_SIZE)

    jepa = build_model(config).to(DEVICE)
    n_params = sum(p.numel() for p in jepa.parameters())
    logger.info(f"Model parameters: {n_params:,}  (n_features={len(SERIES)}, d_model={MODEL_OVERRIDES['d_model']})")

    trainable = list(jepa.encoder.parameters()) + list(jepa.predictor.parameters())
    optimizer = optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    warmup_epochs = max(1, EPOCHS // 10)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                         total_iters=warmup_epochs),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs,
                                                   eta_min=1e-6),
        ],
        milestones=[warmup_epochs],
    )

    jepa.target_encoder.total_steps = len(train_loader) * EPOCHS

    best_val_loss = float("inf")
    log_every = max(1, EPOCHS // 20)

    for epoch in range(EPOCHS):
        jepa.train()
        for batch in train_loader:
            ctx = batch["context"].to(DEVICE)
            tgt = batch["target"].to(DEVICE)
            optimizer.zero_grad()
            loss, _ = jepa.training_step(ctx, tgt)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            jepa.update_target_encoder()
        scheduler.step()

        if (epoch + 1) % log_every == 0 or epoch == EPOCHS - 1:
            val_loss = evaluate(jepa, val_loader)
            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                CKPT_PATH.parent.mkdir(exist_ok=True)
                torch.save({"model": jepa.state_dict(),
                            "n_features": len(SERIES),
                            "epoch": epoch}, CKPT_PATH)
                marker = "  ← best"
            logger.info(f"  Epoch {epoch+1:>3}/{EPOCHS}  val_loss={val_loss:.4f}{marker}")

    logger.info(f"Best val loss: {best_val_loss:.4f}  (checkpoint: {CKPT_PATH})")
    jepa.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE)["model"])
    return jepa


if __name__ == "__main__":
    config = load_config("config/variables.yaml")
    logger.info("Loading cached splits ...")
    splits = build_pipeline(config_path="config/variables.yaml", force_rebuild=False)

    jepa = train(config, splits)

    val_test = subset_panel(
        pd.concat([splits["val"], splits["test"]]).sort_index()
    )
    val_test = val_test[~val_test.index.duplicated(keep="last")]
    logger.info(f"\nEval panel: {val_test.shape}  columns: {list(val_test.columns)}")

    result = run_experiment_5(
        jepa, val_test, config, DEVICE,
        output_dir=Path("results/exp5"),
    )

    print("\n=== Experiment 5 Result (2-series model) ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
