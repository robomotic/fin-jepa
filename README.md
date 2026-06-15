# Financial JEPA World Model

A self-supervised world model for macroeconomics and financial markets, built on the **Joint Embedding Predictive Architecture (JEPA)**. The model learns structural relationships between macroeconomic drivers and asset prices entirely without labels — and is evaluated by proving the latent space has captured meaningful economic geometry.

---

## Why JEPA for Finance?

Standard supervised models predict prices. JEPA predicts **future latent representations** — the model must learn *why* an asset moves, not just *that* it moves. A JEPA trained on macroeconomic and market data should, if it is genuinely learning, encode concepts like "geopolitical risk" and "monetary tightening" as coherent directional vectors in embedding space.

The evaluation suite contains four experiments that test exactly this, without ever using labels or hand-crafted features.

---

## Architecture

```
Context Window [B, T_ctx, D]
        │
        ▼
  PatchEmbed (patch_len=21 days → 1 month per patch)
        │
        ▼
  ContextEncoder E_θ                  TargetEncoder (EMA copy of E_θ)
  (Transformer, 6 layers)             (no gradient — EMA updated only)
        │                                      │
        ▼                                      ▼
  z_ctx [B, N_ctx, d_model]          z_target [B, N_tgt, d_model]
        │
        ▼
  Predictor P_φ (cross-attention, 4 layers)
        │
        ▼
  z_pred [B, N_tgt, d_model]
        │
        ▼
  VICReg Loss (z_pred vs z_target)
  = λ_inv · MSE + λ_var · collapse_penalty + λ_cov · decorrelation
```

**Default patch layout:** 9 context patches (9 months visible) + 3 target patches (predict next 3 months' latents). Total window = 252 trading days (1 year).

**EMA annealing:** τ starts at 0.996, anneals to 0.9999 over training via cosine schedule.

**No labels, no contrastive negatives.** VICReg prevents representational collapse without negative pairs.

---

## Data Suite — 6 Pillars

The 46 series span six macroeconomic pillars. All relationships are regime-conditional (not "ironclad"), and the pillars are intentionally correlated with each other — JEPA must disentangle them.

### Pillar 1 — Cost of Capital

| Series | Source | ID | Role |
|--------|--------|-----|------|
| US 10Y Treasury | FRED | DGS10 | Input |
| US 2Y Treasury | FRED | DGS2 | Input |
| 5Y TIPS Breakeven | FRED | T5YIE | Input |
| 5Y5Y Forward Inflation | FRED | T5YIFR | Input |
| MOVE Index (bond vol) | FRED | MOVE | Input |
| XLK (tech ETF) | Yahoo | XLK | Target |
| XLF (financials ETF) | Yahoo | XLF | Target |
| TLT (20Y+ bonds) | Yahoo | TLT | Both |

Rising real rates compress tech valuations (higher discount rate on future earnings) and can stress financials via duration mismatches. TIPS breakevens separate nominal from real rate effects.

### Pillar 2 — Global Liquidity & Safe Haven

| Series | Source | ID | Role |
|--------|--------|-----|------|
| DXY (US Dollar Index) | Yahoo | DX-Y.NYB | Input |
| VIX (equity vol) | Yahoo | ^VIX | Input |
| EEM (EM equities) | Yahoo | EEM | Both |
| GLD (gold) | Yahoo | GLD | Both |
| FXY (Japanese yen) | Yahoo | FXY | Both |

DXY strength squeezes dollar-denominated EM debt. Gold's relationship with DXY conditions on real rates — when real rates rise alongside DXY, gold is doubly pressured; when DXY rises but real rates are negative, gold can hold up.

### Pillar 3 — Supply Chain & Inflation

| Series | Source | ID | Freq | Role |
|--------|--------|-----|------|------|
| USO (crude oil ETF) | Yahoo | USO | Daily | Both |
| BDI (Baltic Dry Index) | Yahoo | ^BDI | Daily | Input |
| GSCPI | NY Fed | Excel | Monthly (+45d lag) | Input |
| XLY (consumer disc.) | Yahoo | XLY | Daily | Target |
| XLE (energy) | Yahoo | XLE | Daily | Target |
| XLB (materials) | Yahoo | XLB | Daily | Target |

BDI is real-time (no publication lag) unlike CPI. GSCPI decomposes logistics stress from energy-price stress — XLE reacts to energy prices, not supply chain friction, so the distinction matters.

### Pillar 4 — Geopolitical Risk (GPR)

| Series | Source | Download | Freq | Role |
|--------|--------|----------|------|------|
| GPR Global | Caldara & Iacoviello | `gpr_daily_recent.xlsx` | Daily | Input |
| GPRA (Acts) | same | same | Daily | Input |
| GPRT (Threats) | same | same | Daily | Input |
| ITA (defense ETF) | Yahoo | ITA | Daily | Target |

GPR shocks are **exogenous** — wars and terrorist attacks are not caused by financial markets. This makes GPR the best available instrument for pre-registered shock identification in Experiment 2. The GPRA/GPRT split tests whether the model distinguishes *realized events* from *threatening language*.

Monthly file: `https://www.matteoiacoviello.com/gpr_files/gpr_web_latest.xlsx`
Daily file: `https://www.matteoiacoviello.com/gpr_files/gpr_daily_recent.xlsx`

### Pillar 5 — Policy Uncertainty & Financial Conditions

| Series | Source | ID | Freq | Role |
|--------|--------|-----|------|------|
| US EPU | Baker/Bloom/Davis | Excel | Monthly (+30d lag) | Input |
| Global EPU | same | Excel | Monthly (+30d lag) | Input |
| NFCI | FRED | NFCI | Weekly | Input |
| NFCI Risk Sub-index | FRED | NFCIRISK | Weekly | Input |
| STLFSI | FRED | STLFSI4 | Weekly | Input |
| HYG (high yield bonds) | Yahoo | HYG | Daily | Both |
| IWM (small caps) | Yahoo | IWM | Daily | Both |

NFCI and STLFSI are weekly, capturing financial conditions between monthly macro releases. High yield credit spreads (HYG relative to TLT) are the transmission mechanism between policy uncertainty and real investment.

### Pillar 6 — Labor Market & Realized Inflation

| Series | Source | FRED ID | Freq | Pub Lag | Role |
|--------|--------|---------|------|---------|------|
| Core PCE | FRED | PCEPILFE | Monthly | ~30d | Input |
| Core CPI | FRED | CPILFESL | Monthly | ~14d | Input |
| CPI (headline) | FRED | CPIAUCSL | Monthly | ~14d | Input |
| PPI | FRED | PPIACO | Monthly | ~14d | Input |
| UM Inflation Expectations | FRED | MICH | Monthly | ~5d | Input |
| Unemployment Rate | FRED | UNRATE | Monthly | ~5d | Input |
| Non-Farm Payrolls | FRED | PAYEMS | Monthly | ~5d | Input |
| Initial Jobless Claims | FRED | ICSA | **Weekly** | 4d (Thu) | Input |
| JOLTS Job Openings | FRED | JTSJOL | Monthly | ~35d | Input |
| ADP Employment | FRED | ADPMNUSNERSA | Monthly | ~2d | Input |

The Fed's dual mandate (price stability + maximum employment) is what determines rate decisions, which drive Pillar 1. Initial Jobless Claims is the **most timely macro signal in the suite** — published every Thursday for the prior week, 4-day lag.

Publication lag ordering by timeliness:
1. ADP (~2d) → advance NFP signal
2. ICSA (~4d, weekly) → most current labor data
3. MICH / UNRATE / NFP (~5d) → employment Friday
4. CPI / PPI / Core CPI (~14d) → inflation Thursday
5. Core PCE (~30d) → Fed's preferred measure
6. JOLTS (~35d) → longest lag in suite

### Broad Market Regime Trackers (Cross-Pillar)

| Series | Yahoo ID | Role |
|--------|----------|------|
| SPY (S&P 500) | SPY | Both |
| QQQ (NASDAQ 100) | QQQ | Both |
| RSP (Equal-weight S&P) | RSP | Both |
| IWM (Russell 2000) | IWM | Both |
| EFA (Developed ex-US) | EFA | Both |

The **SPY/RSP spread** is a market breadth signal: when SPY rises but RSP lags, gains are concentrated in mega-cap tech (narrow leadership) — a known regime fragility indicator JEPA should learn to detect from macro inputs without being told.

---

## Experiments — Proving the Latent Space Learned

Because JEPA operates in latent space, standard MSE on prices is meaningless. These four experiments test the **geometry of the embedding space** directly.

### Experiment 1 — Linear Probing Test

**The gold standard for self-supervised representations.**

1. Freeze the encoder (zero gradient).
2. Extract patch latents for each window; mean-pool to one vector per window.
3. Train a strictly **linear** head (Ridge regression, no activations) to predict the n-day forward return of a macro-sensitive ratio: `XLK/XLF` (rate regime), `GLD/EEM` (safe haven), `SPY/HYG` (risk-on breadth).
4. Evaluate with **Information Coefficient (Spearman IC)** — the industry standard in quantitative finance.
5. Repeat at 1d, 5d, 20d, 60d horizons.

**Three mandatory baselines:**
| Encoder | What it controls for |
|---------|---------------------|
| Random encoder (never trained) | Floor — establishes chance level |
| Raw features → linear head | Whether JEPA adds over any compression |
| JEPA trained on shuffled sequences | Whether temporal structure is learned |

**The proof:** JEPA IC should exceed all baselines at 20d–60d horizons (macro transmission timescale). If JEPA IC peaks at 1d, it is doing momentum trading, not macro reasoning.

### Experiment 2 — Latent Vector Arithmetic (The Macro Shock Test)

Inspired by Word2Vec arithmetic (`king − man + woman = queen`), but with a critical improvement: **shock periods are defined using GPR, not market-derived signals**, eliminating researcher selection bias.

**Pre-registered shock criterion (set before looking at results):**
- `z̄_shock` = mean embedding when `GPR_DAILY > 90th percentile` of training distribution
- `z̄_calm` = mean embedding when `GPR_DAILY < 25th percentile`
- `v_shock = z̄_shock − z̄_calm`

**GPRA vs GPRT direction test:**
Compute separate vectors for Acts and Threats. Test whether they point in the same direction (cosine similarity > 0.7) but GPRA has a larger L2 norm — indicating the model encodes realized events as a stronger version of the same concept.

**Perturbation test (robustness check):**
Repeat with ±10-percentile threshold shifts. If `v_shock` direction reverses under perturbation, the effect is not robust. Require > 80% of perturbations to be aligned (cosine > 0.5 with base vector).

### Experiment 3 — Context-Masking Verification

Feed the model partial information by **zeroing specific input channels** and measuring whether the latent representation remains structurally consistent.

| Scenario | Visible channels | Expected result |
|----------|-----------------|-----------------|
| `full` | All | Baseline |
| `macro_only` | All non-equity | Latent should stay close to `full` when macro drives regime |
| `yields_only` | Pillar 1 (rates) | IC on QQQ at 20d should stay significant |
| `gpr_only` | Pillar 4 (GPR) | Latent should approach `z̄_shock` centroid on high-GPR days |
| `labor_only` | Pillar 6 (NFP/CPI) | IC on SPY should survive |
| `equity_only` | Yahoo series only | **Falsifiability row: should fail on macro-shock days** |

**MLP baseline:** A small MLP trained only on macro inputs (never sees equity prices) runs the same masking evaluation. If it matches JEPA's masked performance, JEPA is not generalising — it is memorising correlations visible in training.

**Metrics:** cosine similarity of `z_masked` to `z_full` per window, plus IC degradation curves.

### Experiment 4 — Geopolitical Regime Transfer Test (The Hardest Test)

**Out-of-sample shock whose magnitude was never seen in training.**

- **Training data:** 2000–2019 only.
- **Event date:** 2022-02-24 — Russia invades Ukraine. GPR_DAILY spiked to a near-historical record, far above the training distribution.

**Protocol:**
1. Feed the frozen encoder only GPR, DXY, and TIPS breakevens (zero all equity channels).
2. Compute `z_event` = encoder output on the invasion date.
3. Compute `z̄_baseline` = mean encoder output over Jan 2022 (pre-invasion).
4. `Δz = z_event − z̄_baseline`.

**Two measurements:**
- **(a) Cosine similarity** of `Δz` to `v_shock` from Experiment 2. If ≥ 0.5, the model has mapped an out-of-distribution GPR spike onto its existing geopolitical risk vector — it generalised the direction, not just the magnitude.
- **(b) Linear probe scores** from Experiment 1: ITA probe should increase ≥ 2σ above baseline; EEM probe should decrease ≥ 1σ.

**Comparison baseline:** A supervised LSTM trained to predict next-day ITA/EEM returns. The LSTM learned a magnitude-fitted mapping from training data; it saturates on the Feb 24 magnitude. JEPA, having learned directional geometry, should respond proportionally.

---

## Setup

### 1. Clone the repository

```bash
git clone <this-repo>
cd fin-jepa
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Clone TS-JEPA backbone

The project runs with its own self-contained JEPA implementation. To optionally use the TS-JEPA backbone from the NeurIPS Workshop paper:

```bash
git clone https://github.com/Sennadir/TS_JEPA extern/TS_JEPA
pip install -r extern/TS_JEPA/requirements.txt
```

The encoder in [model/encoder.py](model/encoder.py) auto-detects and loads TS-JEPA if `extern/TS_JEPA/` exists. Otherwise it falls back to the built-in causal Transformer implementation.

### 4. Set up FRED API key (required for macro data)

```bash
# Option A: environment variable
export FRED_API_KEY=your_key_here

# Option B: file
echo "your_key_here" > ~/.fred_api_key
```

Free API keys at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html).

---

## Running

### First run — download all data and run diagnostics

```bash
python train.py --epochs 100 --batch-size 64 --device cuda
```

On first run, the pipeline:
1. Downloads all 46 series from FRED, Yahoo Finance, Caldara-Iacoviello, Baker-Bloom-Davis, and NY Fed
2. Applies publication lags (e.g., CPI shifted 14 days forward)
3. Harmonizes to NYSE business day calendar
4. Runs ADF + KPSS stationarity tests, STL decomposition, and ACF analysis per series
5. Applies expanding z-score normalization
6. Caches splits to `data/cache/splits/`

Subsequent runs load from cache (use `--force-rebuild` to re-download).

### Training only (skip experiments)

```bash
python train.py --epochs 200 --batch-size 128 --lr 3e-4 --no-diagnostics
```

### Resume from checkpoint

```bash
python train.py --resume --checkpoint checkpoints/latest.pt --epochs 200
```

### Evaluation only (run experiments on a trained model)

```bash
python train.py --eval-only --checkpoint checkpoints/best.pt
```

### Run individual experiments

```python
from data.pipeline import build_pipeline, load_config
from data.dataset import FinancialJEPADataset
from model.jepa import JEPA, JEPAConfig
import torch

config  = load_config("config/variables.yaml")
splits  = build_pipeline()
# ... load model from checkpoint ...

# Experiment 1
from experiments.exp1_linear_probe import run_experiment_1, print_summary
results = run_experiment_1(jepa, train_ds, test_ds, splits["test"], config, device)
print_summary(results)

# Experiment 2
from experiments.exp2_latent_arithmetic import run_experiment_2
exp2 = run_experiment_2(jepa, train_ds, splits["train"], config, device)

# Experiment 3
from experiments.exp3_context_masking import run_experiment_3
exp3 = run_experiment_3(jepa, train_ds, test_ds, splits["train"], splits["test"],
                         splits["test"], config, device)

# Experiment 4
from experiments.exp4_geopolitical_transfer import run_experiment_4
exp4 = run_experiment_4(jepa, splits["test"], config, device,
                         v_gpr_shock=exp2["v_gpr"])
```

---

## Verification Checklist

### Pipeline

```bash
python -c "
from data.pipeline import build_pipeline
splits = build_pipeline(run_diagnostics=False)
train, val, test = splits['train'], splits['val'], splits['test']
print(train.shape, val.shape, test.shape)
assert train.index.max() < val.index.min(), 'Train/val overlap!'
assert val.index.max()   < test.index.min(), 'Val/test overlap!'
print('Pipeline OK')
"
# Expected: ~5000 × 46 train, ~500 × 46 val, ~750 × 46 test
```

### GPR data

```bash
python -c "
from data.sources.gpr import fetch_gpr_daily
df = fetch_gpr_daily()
print(df.describe())
print(df['2022-02-20':'2022-03-01'])
# Expected: visible spike in GPR_GLOBAL around 2022-02-24
"
```

### Diagnostics

```bash
python -c "
from data.pipeline import build_raw_panel, apply_transforms, load_config
from data.diagnostics import build_diagnostics_report
config = load_config()
panel = build_raw_panel(config)
panel = apply_transforms(panel, config)
report = build_diagnostics_report(panel, config, save_figures=False)
non_stat = [k for k,v in report.items() if v.get('verdict') == 'unit_root']
print('Non-stationary after transform:', non_stat or 'None')
"
```

### Experiment 1 success criteria

- JEPA IC at 20d horizon > Random encoder IC (should be near 0)
- JEPA IC at 20d horizon > Raw-features IC (JEPA adds over compression)
- JEPA IC peak at 20d–60d (macro timescale), not at 1d (momentum)
- Statistical significance: IC p-value < 0.05 on test set

### Experiment 4 success criteria

- `Δz` cosine to `v_shock` ≥ 0.5
- ITA linear probe score on event date ≥ 2σ above Jan 2022 baseline
- EEM linear probe score on event date ≤ 1σ below Jan 2022 baseline
- Supervised LSTM shows no consistent directional response (comparison baseline)

---

## Project Structure

```
fin-jepa/
├── config/
│   └── variables.yaml          # All 46 series: source, transform, pillar, pub_lag
├── data/
│   ├── sources/
│   │   ├── fred.py             # FRED API (yields, VIX, NFCI, CPI, NFP, etc.)
│   │   ├── yahoo.py            # yfinance (ETFs, DXY, BDI)
│   │   ├── gpr.py              # Caldara-Iacoviello GPR daily + monthly Excel
│   │   ├── epu.py              # Baker-Bloom-Davis US + Global EPU Excel
│   │   └── gscpi.py            # NY Fed Global Supply Chain Pressure Index
│   ├── diagnostics.py          # ADF+KPSS, STL decomposition, ACF/PACF, report
│   ├── pipeline.py             # 6-step: download → lag → harmonize → transform
│   │                           #          → diagnostics → z-score → embargo splits
│   └── dataset.py              # PyTorch sliding-window Dataset with pillar masking
├── model/
│   ├── encoder.py              # Patch-embedding causal Transformer + TS-JEPA shim
│   ├── predictor.py            # Cross-attention predictor with learned mask tokens
│   ├── target_encoder.py       # EMA momentum encoder (τ: 0.996 → 0.9999)
│   └── jepa.py                 # VICReg loss + JEPA training step
├── experiments/
│   ├── baselines.py            # Random, raw-features, shuffled-sequence encoders
│   ├── exp1_linear_probe.py    # IC at 4 horizons × 3 pairs × 4 encoders
│   ├── exp2_latent_arithmetic.py  # GPR shock vector, GPRA/GPRT, perturbation test
│   ├── exp3_context_masking.py    # 5 masking scenarios + cosine sim + MLP baseline
│   └── exp4_geopolitical_transfer.py  # Feb 2022 out-of-sample latent shift
├── extern/
│   └── TS_JEPA/                # Optional: git clone Sennadir/TS_JEPA
├── checkpoints/                # Saved model checkpoints (created at runtime)
├── results/                    # Experiment outputs (created at runtime)
│   ├── exp1_ic_results.csv
│   ├── exp2/                   # v_gpr.npy, v_gpra.npy, perturbation.csv
│   ├── exp3/                   # masking.csv
│   └── exp4/                   # delta_z.npy, summary.json
├── data/cache/                 # Parquet caches and diagnostic figures (created at runtime)
├── train.py                    # Entry point: training loop + post-training experiments
├── requirements.txt
└── setup.cfg
```

---

## Design Decisions

### Why not MOMENT or other time-series SSL libraries?

MOMENT uses **masked reconstruction in input space** (predict raw values of masked patches — like BERT for time series). JEPA predicts in **latent space** (predict the target encoder's embedding). These are fundamentally different objectives. MOMENT would destroy the world-model property: a model that reconstructs input values is doing compression, not building an abstract economic state representation.

### Why VICReg instead of contrastive loss?

VICReg (Bardes et al., 2022) prevents representational collapse without requiring negative sample pairs. For financial time series, constructing meaningful negatives is non-trivial (what is the "negative" of a rate-hiking regime?). VICReg's three-term loss — invariance + variance + covariance — achieves the same collapse prevention structurally.

### Why expanding z-score, not rolling?

Rolling normalization with a fixed window leaks future statistics: the window mean/std at time t includes information from t+1, ..., t+window. Expanding normalization uses only history available at t. The 252-day burn-in means the first year of data is excluded from training windows.

### Why GPR for shock identification in Experiment 2?

Market-derived shock identification (e.g., top-quintile VIX days) introduces researcher degrees of freedom: the choice of index, window, and threshold can unconsciously favor the hypothesis. GPR shocks are exogenous — the invasion date is a historical fact, not a market signal. The percentile thresholds are pre-registered before looking at any latent space results.

### Publication lag handling

Monthly series like Core PCE (Fed's preferred inflation gauge) are released ~30 days after the reference month. Feeding October PCE into the model on October 31st would be look-ahead bias — that data is not available until late November. The pipeline shifts each lagged series forward by `pub_lag_days` calendar days **before** forward-filling, ensuring the model only sees what was actually published at each point in time.

---

## References

- Assran et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* (I-JEPA)
- Bardes et al. (2022). *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.*
- Bardes et al. (2024). *Revisiting Feature Prediction for Learning Visual Representations from Video.* (V-JEPA)
- Sennadir et al. (2024). *TS-JEPA: Joint Embedding Goes Temporal.* NeurIPS Workshop on Time Series in the Age of Large Models.
- Caldara & Iacoviello (2022). *Measuring Geopolitical Risk.* American Economic Review.
- Baker, Bloom & Davis (2016). *Measuring Economic Policy Uncertainty.* Quarterly Journal of Economics.
- Benigno et al. (2022). *Global Supply Chain Pressure Index.* NY Fed Liberty Street Economics.
