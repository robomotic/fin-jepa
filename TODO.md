# TODO

## JEPA Architecture Improvements (from OpenAlex literature search, 2026-06-25)

Three new JEPA variants are directly applicable to this project. Listed in recommended priority order.

---

### 1. CF-JEPA — Multi-horizon crops + encoder asymmetry
**Paper:** "CF-JEPA: Mask-free forward prediction with asymmetric encoder utilization for time-series representation learning" (2026)
**Link:** https://arxiv.org/abs/2606.07031

- [ ] Replace mask tokens with random temporal crops as context views
- [ ] Predict at short, mid, and long horizons simultaneously (instead of fixed 63-day target)
- [ ] Ablate downstream tasks: route classification/linear probe to **online encoder**, forecasting/anomaly to **EMA target encoder** — currently all inference uses target encoder only
- [ ] Benchmark MSE reduction on forecasting tasks vs. current VICReg setup

**Why:** Mask-free approach preserves temporal continuity; multi-horizon crops are a drop-in alternative to current patch masking. Encoder asymmetry insight may improve Exp 1 linear probe IC scores.

---

### 2. MTS-JEPA — Multi-resolution prediction + codebook bottleneck
**Paper:** "MTS-JEPA: Multi-Resolution Joint-Embedding Predictive Architecture for Time-Series Anomaly Prediction" (2026)
**Link:** https://arxiv.org/abs/2602.04643

- [ ] Add a second, shorter prediction scale alongside existing 63-day target (e.g., 5-day or 21-day patch)
- [ ] Evaluate soft codebook bottleneck as an alternative or complement to VICReg collapse prevention
- [ ] Test whether multi-resolution objective improves Exp 4 (Ukraine invasion cosine alignment ≥ 0.5)

**Why:** Geopolitical shocks manifest at multiple time scales; multi-resolution encoding should improve both short-term shock detection and long-term regime identification. Codebook is a qualitatively different collapse-prevention mechanism worth ablating against VICReg.

---

### 3. CHARM / Multimodal JEPA — Channel-level text descriptions
**Paper:** "Giving Sensors a Voice: Multimodal JEPA for Semantic Time-Series Embeddings" (2026)
**Link:** https://arxiv.org/abs/2605.31580

- [ ] Add textual channel descriptions (e.g., "US 10Y Treasury yield", "Geopolitical Risk Index") to encoder via gating
- [ ] Map the 6 pillars in `variables.yaml` to semantic text groups for cross-pillar interpretability
- [ ] Evaluate whether text-conditioned gating improves Exp 3 masking scenarios vs. index-based masking

**Why:** Lower priority for raw performance, but the 6-pillar structure in `config/variables.yaml` maps naturally onto text-described channel groups, enabling interpretable inter-pillar relationships.

---

## Notes

- **Finnish trading paper** ("Itseohjautuva JEPA-mallin esikoulutus vahvistusoppimista varten algoritmisessa kaupankäynnissä", 2026): JEPA pretraining + RL for algorithmic trading. No DOI yet — watch for public release.
- **SimMTM** (2023, https://arxiv.org/abs/2302.00861): Manifold-based masked pretraining — useful as an additional competitor baseline for Exp 1 alongside the existing random encoder / raw features / shuffled-sequence baselines.
