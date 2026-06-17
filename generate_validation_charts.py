"""Generate all charts for VALIDATION.md."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from data.pipeline import build_pipeline, load_config
from model.jepa import JEPA, JEPAConfig
from data.dataset import FinancialJEPADataset

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C10Y   = "#1f4e79"   # dark blue  — 10Y
C2Y    = "#c45911"   # burnt orange — 2Y
CSPRD  = "#7030a0"   # purple — spread
CTRAIN = "#2563eb"   # blue — trained model
CRND   = "#dc2626"   # red — random baseline
CGREY  = "#9ca3af"   # grey — reference lines

SPLIT_COLORS = {"train": "#d1fae5", "val": "#fef3c7", "test": "#fee2e2"}

FONT = {"family": "DejaVu Sans"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

SPLIT_BOUNDS = {
    "train": ("1993-01-04", "2019-12-31"),
    "val":   ("2020-02-03", "2021-12-31"),
    "test":  ("2022-01-24", "2024-12-31"),
}


def add_split_bands(ax, alpha: float = 0.12) -> None:
    for name, (s, e) in SPLIT_BOUNDS.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color=SPLIT_COLORS[name], alpha=alpha, zorder=0)


def save(fig, name: str) -> Path:
    p = OUT / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {p}")
    return p


# ─── 1. Raw yield series ──────────────────────────────────────────────────────
def fig1_raw_yields() -> None:
    gs10 = pd.read_parquet("data/cache/fred/DGS10.parquet")["DGS10"].dropna()
    gs2  = pd.read_parquet("data/cache/fred/DGS2.parquet")["DGS2"].dropna()
    mask10 = (gs10.index >= "1993-01-01") & (gs10.index <= "2024-12-31")
    mask2  = (gs2.index  >= "1993-01-01") & (gs2.index  <= "2024-12-31")
    gs10, gs2 = gs10[mask10], gs2[mask2]

    fig, ax = plt.subplots(figsize=(12, 4))
    add_split_bands(ax)
    ax.plot(gs10.index, gs10.values, color=C10Y, lw=1.2, label="10-Year Treasury (GS10)")
    ax.plot(gs2.index,  gs2.values,  color=C2Y,  lw=1.2, label="2-Year Treasury (GS2)", alpha=0.85)
    ax.set_ylabel("Yield (%)")
    ax.set_title("Figure 1 — US Treasury Yields: 10-Year vs 2-Year (1993–2024)", fontweight="bold")
    ax.legend(loc="upper right")

    # Annotate the split regions
    for name, (s, e) in SPLIT_BOUNDS.items():
        mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.97, name.upper(),
                ha="center", va="top", fontsize=8, color="#374151", alpha=0.7)

    save(fig, "fig1_raw_yields.png")


# ─── 2. Yield curve slope (2s10s spread) ─────────────────────────────────────
def fig2_spread() -> None:
    gs10 = pd.read_parquet("data/cache/fred/DGS10.parquet")["DGS10"].dropna()
    gs2  = pd.read_parquet("data/cache/fred/DGS2.parquet")["DGS2"].dropna()
    both = pd.concat([gs10.rename("GS10"), gs2.rename("GS2")], axis=1).dropna()
    both = both[(both.index >= "1993-01-01") & (both.index <= "2024-12-31")]
    spread = both["GS10"] - both["GS2"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    add_split_bands(ax1)
    add_split_bands(ax2)

    ax1.plot(both.index, both["GS10"], color=C10Y, lw=1.2, label="GS10 (10Y)")
    ax1.plot(both.index, both["GS2"],  color=C2Y,  lw=1.2, label="GS2 (2Y)", alpha=0.85)
    ax1.set_ylabel("Yield (%)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title("Figure 2 — Yield Curve Co-movement and Slope (2s10s Spread)", fontweight="bold")

    ax2.fill_between(spread.index, spread.values, 0,
                     where=spread.values >= 0, color=C10Y, alpha=0.35, label="Normal (steep)")
    ax2.fill_between(spread.index, spread.values, 0,
                     where=spread.values < 0,  color=CRND,  alpha=0.40, label="Inverted")
    ax2.axhline(0, color=CGREY, lw=0.8, ls="--")
    ax2.set_ylabel("Spread (pp)")
    ax2.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    save(fig, "fig2_spread.png")


# ─── 3. Scatter: raw yields ───────────────────────────────────────────────────
def fig3_scatter() -> None:
    gs10 = pd.read_parquet("data/cache/fred/DGS10.parquet")["DGS10"].dropna()
    gs2  = pd.read_parquet("data/cache/fred/DGS2.parquet")["DGS2"].dropna()
    both = pd.concat([gs10.rename("GS10"), gs2.rename("GS2")], axis=1).dropna()
    both = both[(both.index >= "1993-01-01") & (both.index <= "2024-12-31")]

    corr = both.corr().loc["GS10", "GS2"]

    fig, ax = plt.subplots(figsize=(5, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(both)))
    ax.scatter(both["GS2"], both["GS10"], c=colors, s=2, alpha=0.5, rasterized=True)

    # OLS line
    m, b = np.polyfit(both["GS2"], both["GS10"], 1)
    x_line = np.linspace(both["GS2"].min(), both["GS2"].max(), 200)
    ax.plot(x_line, m * x_line + b, color=CRND, lw=1.5, ls="--", label=f"OLS  r={corr:.3f}")

    ax.set_xlabel("2-Year Yield, GS2 (%)")
    ax.set_ylabel("10-Year Yield, GS10 (%)")
    ax.set_title("Figure 3 — Raw Yield Co-integration Scatter\n(colour = time, blue→yellow = 1993→2024)",
                 fontweight="bold", fontsize=10)
    ax.legend()
    save(fig, "fig3_scatter.png")


# ─── 4. Transform pipeline ────────────────────────────────────────────────────
def fig4_transforms() -> None:
    gs10_raw = pd.read_parquet("data/cache/fred/DGS10.parquet")["DGS10"].dropna()
    gs2_raw  = pd.read_parquet("data/cache/fred/DGS2.parquet")["DGS2"].dropna()
    both = pd.concat([gs10_raw.rename("GS10"), gs2_raw.rename("GS2")], axis=1).dropna()
    both = both[(both.index >= "1993-01-01") & (both.index <= "2024-12-31")]

    diff_gs10 = both["GS10"].diff().dropna()
    diff_gs2  = both["GS2"].diff().dropna()

    # Expanding z-score (same logic as pipeline)
    def expanding_z(s: pd.Series, min_p: int = 252) -> pd.Series:
        mu  = s.expanding(min_periods=min_p).mean()
        std = s.expanding(min_periods=min_p).std()
        return ((s - mu) / std.clip(lower=1e-8)).clip(-5, 5)

    z_gs10 = expanding_z(both["GS10"])
    z_gs2  = expanding_z(both["GS2"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Figure 4 — Transform Pipeline Applied to Treasury Yields", fontweight="bold", y=1.01)

    # Row 0: raw level
    add_split_bands(axes[0])
    axes[0].plot(both.index, both["GS10"], color=C10Y, lw=1.1, label="GS10 (10Y)")
    axes[0].plot(both.index, both["GS2"],  color=C2Y,  lw=1.1, label="GS2 (2Y)", alpha=0.85)
    axes[0].set_ylabel("Yield (%)")
    axes[0].set_title("Step 1 — Raw Level (as downloaded from FRED)", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=9)

    # Row 1: first difference
    add_split_bands(axes[1])
    axes[1].plot(diff_gs10.index, diff_gs10.values, color=C10Y, lw=0.8, label="Δ GS10")
    axes[1].plot(diff_gs2.index,  diff_gs2.values,  color=C2Y,  lw=0.8, label="Δ GS2", alpha=0.85)
    axes[1].axhline(0, color=CGREY, lw=0.7, ls="--")
    axes[1].set_ylabel("Daily change (pp)")
    axes[1].set_title("Step 2 — First Difference (makes level-stationary series stationary)", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=9)

    # Row 2: expanding z-score (what the model actually sees)
    add_split_bands(axes[2])
    axes[2].plot(z_gs10.index, z_gs10.values, color=C10Y, lw=1.0, label="z(GS10)")
    axes[2].plot(z_gs2.index,  z_gs2.values,  color=C2Y,  lw=1.0, label="z(GS2)", alpha=0.85)
    axes[2].axhline(0, color=CGREY, lw=0.7, ls="--")
    axes[2].set_ylabel("Standard deviations")
    axes[2].set_title("Step 3 — Expanding Z-score (what fin-jepa encoder receives)", fontsize=10)
    axes[2].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    save(fig, "fig4_transforms.png")


# ─── 5 & 6. Exp 5 — cosine sim histogram + time series ───────────────────────
def fig5_fig6_exp5(jepa: JEPA, val_test_panel: pd.DataFrame, config: dict,
                   device: torch.device) -> None:
    model_cfg     = config.get("model", {})
    patch_len     = model_cfg.get("patch_len", 21)
    n_ctx         = model_cfg.get("n_patches_context", 9)
    n_tgt         = model_cfg.get("n_patches_target", 3)

    cols      = list(val_test_panel.columns)
    us10y_idx = cols.index("US10Y")

    ds = FinancialJEPADataset(
        panel=val_test_panel, config=config,
        patch_len=patch_len, n_patches_context=n_ctx, n_patches_target=n_tgt,
        stride=5, masking_strategy="none",
    )

    random_jepa = copy.deepcopy(jepa)
    for m in random_jepa.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    random_jepa.eval().to(device)
    jepa.eval().to(device)

    trained_sims, random_sims, window_dates = [], [], []

    loader = torch.utils.data.DataLoader(
        ds, batch_size=32, shuffle=False,
        collate_fn=lambda ss: {
            "context": torch.stack([s["context"] for s in ss]),
            "target":  torch.stack([s["target"]  for s in ss]),
            "meta":    [s["meta"] for s in ss],
        },
    )

    def cosine(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        a = a.mean(1); b = b.mean(1)
        return ((a / (a.norm(dim=1, keepdim=True) + 1e-8)) *
                (b / (b.norm(dim=1, keepdim=True) + 1e-8))).sum(1).cpu().numpy()

    with torch.no_grad():
        for batch in loader:
            ctx = batch["context"].to(device)
            tgt = batch["target"].to(device)
            ctx_m = ctx.clone(); ctx_m[:, :, us10y_idx] = 0.0

            zp, zt = jepa(ctx_m, tgt)
            trained_sims.extend(cosine(zp, zt).tolist())

            zpr, ztr = random_jepa(ctx_m, tgt)
            random_sims.extend(cosine(zpr, ztr).tolist())

            for meta in batch["meta"]:
                window_dates.append(pd.Timestamp(meta.context_end_date))

    trained_arr = np.array(trained_sims)
    random_arr  = np.array(random_sims)
    dates_arr   = pd.DatetimeIndex(window_dates)

    # Fig 5: histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(-0.5, 1.0, 40)
    ax.hist(random_arr,  bins=bins, color=CRND,   alpha=0.55, label=f"Random weights  (μ={random_arr.mean():.3f})")
    ax.hist(trained_arr, bins=bins, color=CTRAIN,  alpha=0.65, label=f"Trained model   (μ={trained_arr.mean():.3f})")
    ax.axvline(0.30, color="black", lw=1.2, ls="--", label="Pass threshold (0.30)")
    ax.set_xlabel("Cosine similarity (predicted vs target latent)")
    ax.set_ylabel("Window count")
    ax.set_title("Figure 5 — Exp 5: Distribution of Cosine Similarities\n"
                 "(US10Y masked from context; US02Y visible)", fontweight="bold")
    ax.legend()
    save(fig, "fig5_exp5_histogram.png")

    # Fig 6: time series of cosine sim
    order = np.argsort(dates_arr)
    d_sorted = dates_arr[order]
    t_sorted = trained_arr[order]
    r_sorted = random_arr[order]

    # 21-day rolling mean
    roll_t = pd.Series(t_sorted, index=d_sorted).rolling(21, min_periods=1).mean()
    roll_r = pd.Series(r_sorted, index=d_sorted).rolling(21, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    add_split_bands(ax, alpha=0.15)
    ax.scatter(d_sorted, t_sorted, color=CTRAIN, s=6, alpha=0.30, zorder=2)
    ax.plot(roll_t.index, roll_t.values, color=CTRAIN, lw=1.6,
            label=f"Trained (21d rolling mean, overall μ={trained_arr.mean():.3f})")
    ax.plot(roll_r.index, roll_r.values, color=CRND, lw=1.2, ls="--",
            label=f"Random baseline (μ={random_arr.mean():.3f})", alpha=0.7)
    ax.axhline(0.30, color="black", lw=1.0, ls=":", label="Pass threshold (0.30)")
    ax.axhline(0.00, color=CGREY, lw=0.7)
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Figure 6 — Exp 5: Cosine Similarity Over Time\n"
                 "(US10Y masked from context; predicted vs full-information target)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "fig6_exp5_timeseries.png")


# ─── 7. Exp 1 — Linear probe IC bar chart ────────────────────────────────────
def fig7_exp1_ic() -> None:
    df = pd.read_csv("results/exp1_ic_results.csv")
    jepa_df = df[df["encoder"] == "JEPA"]

    pairs    = jepa_df["target_pair"].unique()
    horizons = sorted(jepa_df["horizon_days"].unique())
    x        = np.arange(len(horizons))
    width    = 0.22
    colours  = [C10Y, C2Y, CSPRD]

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, pair in enumerate(pairs):
        sub = jepa_df[jepa_df["target_pair"] == pair].set_index("horizon_days").reindex(horizons)
        ax.bar(x + i * width, sub["IC"].values, width=width * 0.9,
               color=colours[i % len(colours)], label=pair, alpha=0.85)

    ax.axhline(0, color=CGREY, lw=0.8, ls="--")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Spearman IC")
    ax.set_title("Figure 7 — Exp 1: Linear Probe IC Across Target Pairs & Horizons\n"
                 "(frozen JEPA encoder → Ridge → Spearman rank correlation)", fontweight="bold")
    ax.legend(title="Target pair", fontsize=9)
    save(fig, "fig7_exp1_ic.png")


# ─── 8. Exp 3 — Context masking scenarios ────────────────────────────────────
def fig8_exp3_masking() -> None:
    df = pd.read_csv("results/exp3/exp3_masking.csv")

    # Exclude MLP baseline from IC bar (different model class)
    main = df[~df["scenario"].str.startswith("MLP")].copy()
    mlp  = df[df["scenario"].str.startswith("MLP")].copy()

    colours = [CTRAIN if s == "full" else
               CRND   if s == "equity_only" else
               C2Y    for s in main["scenario"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # IC
    bars = ax1.barh(main["scenario"], main["IC"], color=colours, alpha=0.85)
    if not mlp.empty:
        for _, row in mlp.iterrows():
            ax1.axvline(row["IC"], color="#059669", lw=1.5, ls="--",
                        label=f"{row['scenario']} ({row['IC']:.3f})")
    ax1.axvline(0, color=CGREY, lw=0.8, ls=":")
    ax1.set_xlabel("Spearman IC (20-day horizon)")
    ax1.set_title("Exp 3: IC by masking scenario", fontweight="bold")
    ax1.legend(fontsize=8)

    # Cosine similarity to full-context latent
    main_cos = main.dropna(subset=["cosine_sim_to_full"])
    cos_colours = [CTRAIN if s == "full" else
                   CRND   if s == "equity_only" else
                   C2Y    for s in main_cos["scenario"]]
    ax2.barh(main_cos["scenario"], main_cos["cosine_sim_to_full"], color=cos_colours, alpha=0.85)
    ax2.axvline(1.0, color=CGREY, lw=0.8, ls=":")
    ax2.set_xlabel("Cosine similarity to full-context latent")
    ax2.set_title("Latent alignment when information is removed", fontweight="bold")

    fig.suptitle("Figure 8 — Exp 3: What Happens When We Mask Input Pillars",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig8_exp3_masking.png")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)

    logger.info("Generating Part 1 charts (yield curve correlation) ...")
    fig1_raw_yields()
    fig2_spread()
    fig3_scatter()

    logger.info("Generating Part 2 charts (transform pipeline) ...")
    fig4_transforms()

    logger.info("Loading model + splits for Exp 5 charts ...")
    config = load_config("config/variables.yaml")
    splits = build_pipeline(config_path="config/variables.yaml", force_rebuild=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load("checkpoints/best.pt", map_location=device)
    jepa_cfg = JEPAConfig(
        n_features=ckpt["n_features"],
        **{k: v for k, v in ckpt["config"].items() if k in JEPAConfig.__dataclass_fields__},
    )
    jepa = JEPA(jepa_cfg).to(device)
    jepa.load_state_dict(ckpt["model"])

    val_test = pd.concat([splits["val"], splits["test"]]).sort_index()
    val_test = val_test[~val_test.index.duplicated(keep="last")]

    logger.info("Generating Part 3 charts (JEPA evidence) ...")
    fig5_fig6_exp5(jepa, val_test, config, device)
    fig7_exp1_ic()
    fig8_exp3_masking()

    logger.info(f"All charts written to {OUT}/")
