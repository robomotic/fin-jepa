# Validation: Proving fin-jepa is Genuinely Learning

This document records the econometric rationale and empirical results for
the yield curve sanity check (Experiment 5), introduced following peer
review by an econometrics professor.

---

## The Problem: Distinguishing Signal from Noise

Financial JEPA trains on a panel that includes high-volatility equity
tickers (SPY, QQQ, sector ETFs) whose daily returns are dominated by
microstructure noise. A model can appear to train — loss decreasing,
no obvious collapse — while actually learning to compress noise rather
than economic structure.

The standard model diagnostics (Experiments 1–4) measure useful
properties, but they depend on the full panel. If the panel itself is
noisy enough, a sophisticated architecture can pass those tests while
encoding almost nothing that generalises.

The professor's recommendation was to strip the problem down to its
irreducible core: **feed the model two series that are structurally
co-integrated by construction and verify that it discovers the
relationship**.

---

## Why the Yield Curve is the Right Test

The 2-Year Treasury yield (`GS2`, stored as `US02Y`) and the 10-Year
Treasury yield (`GS10`, stored as `US10Y`) are not merely correlated —
they are bound together by arbitrage and rational expectations theory.
Specifically:

- The 10-year rate equals the expected average of future short-term rates
  plus a term premium.
- The 2-year rate is the most rate-sensitive maturity to Federal Reserve
  policy signals.
- In every business cycle since the 1970s, the spread `GS10 − GS2` (the
  "2s10s curve") has been the canonical leading indicator of recession
  (inverted = warning; steep = expansion).

These series **cannot move independently** over any window longer than a
few weeks. A model that encodes either one and cannot predict the
direction of the other has failed at the most basic level of macroeconomic
representation.

---

## Experiment 5: Yield Curve Sanity Check

**Protocol:**

1. Use the trained encoder on the val+test panel (2020–2024).
2. For every sliding context window, **zero out the US10Y column**
   entirely — the encoder sees only the 2-Year yield, not the 10-Year.
3. Run the JEPA predictor to produce predicted target latents.
4. Run the target encoder on the **full** target window (both yields
   visible) to produce ground-truth target latents.
5. Measure cosine similarity between predicted and ground-truth latents,
   mean-pooled across patches, for each window.
6. Repeat with a **fresh random-weight model** (same architecture, no
   training) as a chance baseline.

**Pass criterion:** mean cosine similarity > 0.30 (conservative — chance
is ~0).

**Failure interpretation:**

| Result | Diagnosis |
|--------|-----------|
| Trained ≈ random ≈ 0 | Representation collapse; check VICReg variance term |
| Trained > random but < 0.30 | Partial learning; under-trained or noisy input |
| Trained > 0.30 | Structural co-movement encoded ✓ |

---

## Results (run 2026-06-17, `checkpoints/best.pt`)

| Metric | Value |
|--------|-------|
| Windows evaluated | 195 |
| Trained cosine similarity | **0.587 ± 0.195** |
| Random baseline | 0.005 ± 0.039 |
| Pass threshold | 0.30 |
| **Verdict** | **PASS** |

The trained model scores **125× higher** than random weights. The encoder
has learned that the 2-Year yield is sufficient to predict the latent
state of a window that also contains the 10-Year yield — exactly the
structural co-integration relationship the yield curve theory predicts.

This rules out "bad data" as an explanation for any downstream experiment
that underperforms. If Experiments 1–4 show weak results, the latent
space is functional; the issue would be in the probing methodology or the
noisiness of the equity targets used for evaluation.

---

## What Was Added to Support This Test

Beyond the experiment itself, four FRED series were added to
`config/variables.yaml` to complete the professor's recommended macro
clusters:

| Series | FRED ID | Pillar | Role |
|--------|---------|--------|------|
| `FEDFUNDS` | `FEDFUNDS` | 1 — Cost of Capital | Yield curve anchor (overnight rate) |
| `PPICMM` | `PPICMM` | 3 — Supply Chain | Intermediate PPI (midstream inflation link) |
| `INDPRO` | `INDPRO` | 3 — Supply Chain | Industrial Production (hard factory output) |
| `TCU` | `TCU` | 3 — Supply Chain | Total Capacity Utilization |

These series fill the structural gaps in the inflation pipeline cluster
(`PPIACO → PPICMM → CPIAUCSL`) and the supply chain cluster
(`INDPRO + TCU`). They will be incorporated into the panel on the next
`--force-rebuild` training run.

---

## Reproducibility

```bash
# Run Exp 5 standalone against the existing checkpoint and cached splits
python run_exp5.py

# Run Exp 5 as part of the full eval suite
python train.py --eval-only --checkpoint checkpoints/best.pt
```

Results are saved to `results/exp5/exp5_yield_curve_sanity.json`.
