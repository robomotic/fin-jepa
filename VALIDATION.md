# Validation: Experiment 5 — Yield Curve Sanity Check

## Motivation

A JEPA model can show decreasing loss while actually compressing noise
from high-volatility equity tickers rather than discovering genuine
economic structure.  To rule this out definitively, we run a minimal
isolation test:

> **Train on only two series.  Mask one entirely from the context.
> Check whether the model can still predict the other.**

If JEPA passes, the architecture is genuinely learning co-movement.
If it fails, the problem is in the model — not the data.

---

## Why US10Y and US02Y?

The 10-Year Treasury yield (`GS10`, stored as `US10Y`) and the 2-Year
Treasury yield (`GS2`, stored as `US02Y`) are structurally co-integrated
by the expectations theory of the term structure:

> GS10 ≈ expected average of future short rates over 10 years + term premium

The 2-Year closely tracks Federal Reserve policy signals; the 10-Year
reflects the long-run growth and inflation outlook.  Both are driven by
the same macroeconomic forces.  Over 30 years of daily data they form a
tight linear manifold (Pearson r ≈ 0.97):

![Figure 1 — Raw Treasury yields (1993–2024)](docs/figures/fig1_raw_yields.png)

The spread between them (GS10 − GS2, the "2s10s curve") is the canonical
business-cycle indicator.  Inversions reliably precede recessions by
12–18 months:

![Figure 2 — 2s10s yield curve slope](docs/figures/fig2_spread.png)

The co-integration holds across all absolute rate levels — whether yields
are at 8% or near zero:

![Figure 3 — Co-integration scatter (colour = time, blue→yellow = 1993→2024)](docs/figures/fig3_scatter.png)

---

## What the Model Sees

Raw yields are non-stationary level series.  The pipeline applies three
steps before the encoder receives any data:

1. Per-series transform (`level` for both yields in this experiment).
2. Reindexed to NYSE business days; gaps forward-filled.
3. **Expanding z-score** (252-day burn-in, clipped at ±5 σ) — never
   rolling, to prevent future-statistic leakage.

The bottom panel below is what the encoder actually processes:

![Figure 4 — Transform pipeline: raw → diff → expanding z-score](docs/figures/fig4_transforms.png)

---

## Protocol

1. Extract only `US10Y` and `US02Y` from the cached panel (2 columns).
2. Train a **fresh JEPA from scratch** on these 2 series only
   (`n_features=2`, `d_model=128`, 4 encoder layers, 200 epochs on
   the 1993–2019 training split — 1,269 windows, ~2.5 minutes on GPU).
3. Evaluate on the combined val + test panel (2020–2024, 195 windows).
4. For each context window, **zero out `US10Y` entirely** — the encoder
   sees only the 2-Year yield.
5. Run the predictor to produce predicted target latents.
6. Run the target encoder on the **full** (unmasked) target to produce
   ground-truth latents.
7. Compute cosine similarity between predicted and ground-truth latents,
   mean-pooled across the 3 target patches.
8. Repeat with **fresh random weights** as a chance baseline.

**Pass criterion:** mean cosine similarity > 0.30 (chance ≈ 0).

**Failure interpretation:**

| Outcome | Diagnosis |
|---------|-----------|
| Trained ≈ random ≈ 0 | Representation collapse — inspect VICReg variance term |
| Trained > random but < 0.30 | Partial learning — under-trained or signal too noisy |
| Trained > 0.30 | Structural co-movement encoded ✓ |

---

## Results

**2-series model trained from scratch, evaluated 2026-06-17:**

| Metric | Value |
|--------|-------|
| Training windows | 1,269 (1993–2019, US10Y + US02Y only) |
| Evaluation windows | 195 (2020–2024) |
| Trained cosine similarity | **0.610 ± 0.195** |
| Random-weight baseline | −0.127 ± 0.065 |
| Pass threshold | 0.30 |
| **Verdict** | **PASS** |

The trained model's distribution is entirely separated from the random
baseline (which is negative — untrained weights actively anti-correlate
the predictions):

![Figure 5 — Cosine similarity distribution: trained vs random](docs/figures/fig5_exp5_histogram.png)

The result holds across the full 2020–2024 evaluation window, including
the 2022 rate hiking cycle and the 2023–24 inversion:

![Figure 6 — Cosine similarity over time (21-day rolling mean)](docs/figures/fig6_exp5_timeseries.png)

---

## Conclusion

A JEPA model trained on **only two yield series** — with no equity
tickers, no macro indicators, no cross-pillar context — correctly
predicts the latent state of the 10-Year yield from the 2-Year yield
alone, scoring **0.610** against a random baseline of **−0.127**.

This rules out "bad data" as an explanation for any downstream result
that underperforms.  The model is learning.

---

## Reproducibility

```bash
# Train 2-series model and run Exp 5 (uses cached splits, ~2.5 min on GPU)
python run_exp5.py

# Regenerate charts
python generate_validation_charts.py
```

Results saved to `results/exp5/exp5_yield_curve_sanity.json`.
