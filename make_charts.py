"""Generate charts for REPORT.md and tweets.md from existing results."""

import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

CHARTS = Path("charts")
CHARTS.mkdir(exist_ok=True)

BLUE   = "#2563EB"
ORANGE = "#EA580C"
GREEN  = "#16A34A"
RED    = "#DC2626"
GREY   = "#6B7280"
LIGHT  = "#F3F4F6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})


# ── 1. Training curve ─────────────────────────────────────────────────────────

log_file = Path("/tmp/retrain_run3.log")
epochs, train_losses, val_losses = [], [], []
pattern = re.compile(
    r"Epoch (\d+)/100.*?train_loss=([\d.]+).*?val_loss=([\d.]+)"
)
with open(log_file) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
            val_losses.append(float(m.group(3)))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, train_losses, color=BLUE,   lw=2, label="Train loss")
ax.plot(epochs, val_losses,   color=ORANGE, lw=2, label="Val loss")

best_epoch = epochs[int(np.argmin(val_losses))]
best_val   = min(val_losses)
ax.axvline(best_epoch, color=GREEN, lw=1.5, ls=":", label=f"Best checkpoint (epoch {best_epoch})")
ax.scatter([best_epoch], [best_val], color=GREEN, zorder=5, s=60)
ax.axvspan(0, 10, alpha=0.08, color=BLUE, label="Warmup period (10 epochs)")

ax.set_xlabel("Epoch")
ax.set_ylabel("VICReg loss")
ax.set_title("Training curve", fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(1, 100)

fig.tight_layout()
fig.savefig(CHARTS / "training_curve.png")
plt.close(fig)
print("Saved training_curve.png")


# ── 2. IC comparison (grouped bar) ────────────────────────────────────────────

ic_df = pd.read_csv("results/exp1_ic_results.csv")

pairs    = ["XLK/XLF", "GLD/EEM", "SPY/HYG"]
horizons = [1, 5, 20, 60]
encoders = ["JEPA", "Random", "RawFeatures"]
colors   = {
    "JEPA":        BLUE,
    "Random":      GREY,
    "RawFeatures": ORANGE,
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

for ax, pair in zip(axes, pairs):
    sub = ic_df[ic_df["target_pair"] == pair]
    x   = np.arange(len(horizons))
    w   = 0.25
    for idx, enc in enumerate(encoders):
        row = sub[sub["encoder"] == enc].set_index("horizon_days")["IC"]
        vals = [row.get(h, np.nan) for h in horizons]
        bars = ax.bar(x + (idx - 1) * w, vals, w,
                      color=colors[enc], alpha=0.85, label=enc)
        # value labels on bars
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ypos = bar.get_height() if v >= 0 else bar.get_height() - 0.02
                ax.text(bar.get_x() + bar.get_width() / 2,
                        ypos + (0.005 if v >= 0 else -0.02),
                        f"{v:+.2f}", ha="center", va="bottom",
                        fontsize=7.5, color="black")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(pair, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.set_xlabel("Forecast horizon")

axes[0].set_ylabel("Spearman IC")

patches = [mpatches.Patch(color=colors[e], label=e) for e in encoders]
fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Experiment 1: Spearman IC by horizon and asset pair",
             fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(CHARTS / "exp1_ic_comparison.png", bbox_inches="tight")
plt.close(fig)
print("Saved exp1_ic_comparison.png")


# ── 3. Exp 2: perturbation robustness ─────────────────────────────────────────

pert = pd.read_csv("results/exp2/exp2_perturbation.csv")
labels = [f"p{int(r.shock_pct)}/p{int(r.calm_pct)}" for _, r in pert.iterrows()]
cosines = pert["cosine_to_base"].tolist()
is_robust = pert["robust"].tolist()

bar_colors = [GREEN if r else RED for r in is_robust]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(labels, cosines, color=bar_colors, alpha=0.85, height=0.55)
ax.axvline(0.5, color=RED, lw=1.5, ls="--", label="Robustness threshold (0.5)")
ax.axvline(1.0, color=GREY, lw=0.8, ls=":")

for bar, v in zip(bars, cosines):
    ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", fontsize=9)

ax.set_xlim(0, 1.08)
ax.set_xlabel("Cosine similarity to base shock vector (p90/p25)")
ax.set_title("Experiment 2: Shock vector stability across GPR thresholds",
             fontweight="bold")
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(CHARTS / "exp2_perturbation.png")
plt.close(fig)
print("Saved exp2_perturbation.png")


# ── 4. Exp 3: context masking ─────────────────────────────────────────────────

mask_df = pd.read_csv("results/exp3/exp3_masking.csv")
mask_df = mask_df[mask_df["scenario"] != "MLP_baseline_macro_only"].copy()

scenario_labels = {
    "full":        "Full (44ch)",
    "macro_only":  "Macro only",
    "yields_only": "Yields only",
    "gpr_only":    "GPR only",
    "labor_only":  "Labour only",
    "equity_only": "Equity only\n(falsif. check)",
}
mask_df["label"] = mask_df["scenario"].map(scenario_labels)

mlp_ic = pd.read_csv("results/exp3/exp3_masking.csv")
mlp_ic = mlp_ic[mlp_ic["scenario"] == "MLP_baseline_macro_only"]["IC"].values[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A: cosine to full
cos_colors = []
for sc in mask_df["scenario"]:
    if sc == "full":
        cos_colors.append(BLUE)
    elif sc == "equity_only":
        cos_colors.append(RED)
    else:
        cos_colors.append(ORANGE)

bars = ax1.barh(mask_df["label"], mask_df["cosine_sim_to_full"],
                color=cos_colors, alpha=0.85, height=0.55)
ax1.axvline(0, color="black", lw=0.8)
for bar, v in zip(bars, mask_df["cosine_sim_to_full"]):
    xpos = v + 0.02 if v >= 0 else v - 0.02
    ha   = "left" if v >= 0 else "right"
    ax1.text(xpos, bar.get_y() + bar.get_height() / 2,
             f"{v:.2f}", va="center", fontsize=9, ha=ha)
ax1.set_xlabel("Cosine similarity to full representation")
ax1.set_title("Representational similarity\n(higher = masks less structure)",
              fontweight="bold")
ax1.set_xlim(-0.3, 1.2)

# Panel B: IC
ic_colors = []
for sc in mask_df["scenario"]:
    if sc == "equity_only":
        ic_colors.append(RED)
    elif sc == "labor_only":
        ic_colors.append(GREEN)
    else:
        ic_colors.append(BLUE)

bars2 = ax2.barh(mask_df["label"], mask_df["IC"],
                 color=ic_colors, alpha=0.85, height=0.55)
ax2.axvline(0, color="black", lw=0.8)
ax2.axvline(mlp_ic, color=GREY, lw=1.5, ls="--",
            label=f"MLP macro baseline (IC={mlp_ic:.3f})")
for bar, v in zip(bars2, mask_df["IC"]):
    xpos = v + 0.005 if v >= 0 else v - 0.005
    ha   = "left" if v >= 0 else "right"
    ax2.text(xpos, bar.get_y() + bar.get_height() / 2,
             f"{v:+.3f}", va="center", fontsize=9, ha=ha)
ax2.set_xlabel("Spearman IC (20d XLK/XLF)")
ax2.set_title("Predictive IC by masking scenario",
              fontweight="bold")
ax2.legend(fontsize=9)

fig.suptitle("Experiment 3: Context masking", fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(CHARTS / "exp3_masking.png", bbox_inches="tight")
plt.close(fig)
print("Saved exp3_masking.png")


# ── 5. Exp 4: Ukraine event detection ────────────────────────────────────────

z_event    = np.load("results/exp4/exp4_z_event.npy")
z_baseline = np.load("results/exp4/exp4_z_baseline.npy")
delta_z    = np.load("results/exp4/exp4_delta_z.npy")
v_gpr      = np.load("results/exp2/v_gpr.npy")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: distribution of |delta_z| vs baseline noise
ax = axes[0]
ax.hist(delta_z, bins=40, color=RED, alpha=0.7, label="Dz (event - baseline)")
ax.axvline(0, color="black", lw=0.8)
ax.axvline(np.linalg.norm(delta_z), color=RED, lw=0, alpha=0)  # just for annotation
ax.set_xlabel("Latent dimension value")
ax.set_ylabel("Count")
ax.set_title("Distribution of Dz (Ukraine event minus Jan 2022 baseline)",
             fontweight="bold")
# Annotate norm
ax.text(0.97, 0.93, f"||Dz|| = {np.linalg.norm(delta_z):.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec=GREY))

# Panel B: cosine landscape — bar for each dimension contribution
ax = axes[1]
cos_to_shock = float(
    np.dot(delta_z, v_gpr) /
    (np.linalg.norm(delta_z) * np.linalg.norm(v_gpr) + 1e-8)
)

# Show top-30 dimensions by |delta_z| magnitude to give a sense of the shift
top_idx  = np.argsort(np.abs(delta_z))[-30:][::-1]
dim_vals = delta_z[top_idx]
dim_cols = [GREEN if v > 0 else RED for v in dim_vals]

bars = ax.bar(range(len(top_idx)), dim_vals, color=dim_cols, alpha=0.75, width=0.7)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Latent dimension (top 30 by |Dz|)")
ax.set_ylabel("Dz value")
ax.set_title("Top-30 shifted latent dimensions on 2022-02-24",
             fontweight="bold")
ax.set_xticks([])
ax.text(0.97, 0.07,
        f"cos(Dz, v_shock) = {cos_to_shock:+.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec=GREY))

pos_patch = mpatches.Patch(color=GREEN, alpha=0.75, label="Positive shift")
neg_patch = mpatches.Patch(color=RED,   alpha=0.75, label="Negative shift")
ax.legend(handles=[pos_patch, neg_patch], fontsize=9)

fig.suptitle("Experiment 4: Ukraine invasion latent shift (2022-02-24)",
             fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(CHARTS / "exp4_ukraine.png", bbox_inches="tight")
plt.close(fig)
print("Saved exp4_ukraine.png")

print("\nAll charts saved to charts/")
