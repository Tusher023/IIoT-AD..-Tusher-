# Research Questions (Revised — Aligned with Evidence-Based Gap)

## Primary Research Question

> **To what extent do unsupervised anomaly-detection methods differ in their ability to identify machine degradation and provide early failure warnings in multivariate industrial sensor data, when evaluated under a unified, methodologically rigorous framework with strict leakage prevention?**

*Note: This is an empirical comparison question, not a "does our new method beat the baseline" question. The answer may be that simpler methods perform comparably — that is a valid and valuable finding.*

---

## Secondary Research Questions

### RQ1: Baseline vs. Deep Learning Under Fair Conditions
> When Isolation Forest and Autoencoder-based methods are evaluated under identical data splits, normalization, threshold selection, and metrics — how do their detection performances compare?

**Why this matters:** Many papers compare methods under inconsistent experimental conditions. Fair comparison may reveal that performance differences are smaller than typically reported.

**Evidence needed:** Performance table with identical experimental setup, statistical significance tests across 5 seeds.

---

### RQ2: Does Temporal Modeling Improve Detection?
> Does an LSTM Autoencoder, which explicitly models temporal dependencies in sensor sequences, capture degradation patterns more effectively than a conventional feedforward Autoencoder that treats each timestep independently?

**Why this matters:** LSTM-AE adds substantial computational overhead. If it doesn't meaningfully improve detection, practitioners should know.

**Evidence needed:** Performance comparison + anomaly-score trajectory visualization for representative engines + ablation on sequence length.

---

### RQ3: How Does Detection Behave Across Operating Conditions?
> How does the performance of each method change when the dataset includes multiple operating conditions (FD001 vs FD002) or multiple fault modes (FD001 vs FD003)?

**Why this matters:** Real industrial systems operate under varying conditions. A model that only works under one condition has limited practical value.

**Evidence needed:** Per-subset performance table + cross-subset evaluation (train FD001 → test FD002) + performance degradation quantification.

---

### RQ4: How Early Can Each Method Detect Degradation?
> For each method, how many cycles before failure does the first sustained anomaly warning appear, and how does this vary across engines?

**Why this matters:** An anomaly detector that only triggers 5 cycles before failure offers less value than one that triggers 50 cycles early. Mean lead time is more actionable than F1 alone.

**Evidence needed:** Lead-time distribution (boxplot/histogram) per method + per-engine detection timeline + missed failure analysis.

---

### RQ5: What Is the Sensitivity–False Alarm Trade-off?
> How does the choice of anomaly threshold affect the trade-off between detection sensitivity and false alarm rate for each method?

**Why this matters:** In real systems, false alarms are costly. A method with high recall but unacceptable false alarm rate may be worse than one with moderate recall and low false alarms.

**Evidence needed:** Precision-recall curves + threshold sweep analysis + operating-point recommendation under different cost assumptions.

---

### RQ6: Do Models Generalize to Unseen Engines?
> When models are trained on one set of engines and evaluated on completely separate engines (engine-level test split), how much does performance degrade compared to evaluation on training engines?

**Why this matters:** Tests whether the model has learned general degradation patterns vs. memorizing specific engine behavior.

**Evidence needed:** Generalization gap table (train-set performance vs. test-set performance) + analysis of which engine characteristics predict poor generalization.

---

### RQ7: Is Added Complexity Justified?
> Does the marginal performance improvement from more complex models (LSTM-AE > AE > IF > statistical threshold) justify the additional computational cost?

**Why this matters:** For resource-constrained edge deployment or rapid prototyping, practitioners need to know the cost-benefit trade-off.

**Evidence needed:** Performance-vs-complexity chart (F1 vs. parameter count, F1 vs. training time) + concrete recommendation table.

---

## Research Question → Experiment Mapping

| RQ | Primary Experiment | Dataset(s) | Key Metrics | Ablation |
|----|-------------------|------------|-------------|----------|
| RQ1 | Fair model comparison | FD001 | F1, PR-AUC, Lead Time | Threshold strategy |
| RQ2 | Temporal vs non-temporal | FD001 | F1, Anomaly trajectories | Sequence length |
| RQ3 | Cross-condition evaluation | FD001→FD002, FD003, FD004 | Per-condition F1 | Op. condition normalization |
| RQ4 | Early warning analysis | FD001 | Lead time distribution | Persistence window |
| RQ5 | Threshold sensitivity | FD001 | PR curves, FAR | Multiple threshold strategies |
| RQ6 | Generalization test | FD001 | Train vs test gap | Engine subset size |
| RQ7 | Complexity analysis | FD001 | F1 / training time ratio | Model size variants |

---

## Guiding Principle

> We do not predetermine which method will "win." If Isolation Forest outperforms LSTM-AE under certain conditions, that is a finding we report with the same rigor as the reverse. The goal is understanding, not advocacy.
