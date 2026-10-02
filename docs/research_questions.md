# Research Questions

## Primary Research Question

> **How effectively can unsupervised anomaly-detection methods identify machine degradation and provide early warnings of impending failure in multivariate industrial sensor data when labeled failure examples are limited?**

---

## Secondary Research Questions

### RQ1: Classical vs. Deep Learning Comparison
> How does Isolation Forest compare with Autoencoder-based anomaly detection for identifying machine degradation?

**Experimental approach:**
- Train Isolation Forest and Autoencoder on same data splits
- Compare using precision, recall, F1, PR-AUC, detection lead time
- Control for threshold selection strategy
- Repeat across multiple seeds

**Expected evidence:** Quantitative performance comparison table + statistical significance test

---

### RQ2: Temporal vs. Non-Temporal Modeling
> Does an LSTM Autoencoder capture temporal degradation patterns better than a conventional Autoencoder?

**Experimental approach:**
- Compare FC-Autoencoder (per-timestep) vs. LSTM-AE (sequence-based)
- Analyze whether sequence-level reconstruction captures gradual degradation
- Compare anomaly score trajectories for representative engines
- Ablation: vary sequence length

**Expected evidence:** Performance metrics + anomaly-score trajectory comparison + ablation table

---

### RQ3: Operating Condition Sensitivity
> How does model performance change under different operating conditions?

**Experimental approach:**
- FD001 (1 condition) vs. FD002 (6 conditions)
- Train under one condition set, evaluate under another
- Analyze per-condition performance breakdown

**Expected evidence:** Cross-condition performance matrix + degradation analysis

---

### RQ4: Early Warning Capability
> How early can each method detect abnormal machine behavior before actual failure?

**Experimental approach:**
- Define early-warning framework (sustained anomaly score above threshold)
- Compute first-detection time for each engine
- Compute lead time (cycles before failure)
- Compare across models

**Expected evidence:** Lead-time distribution + mean/median lead time table + per-engine timelines

---

### RQ5: Sensitivity vs. False Alarm Trade-off
> What is the trade-off between detection sensitivity and false alarms?

**Experimental approach:**
- Vary threshold across range
- Plot precision-recall curves
- Analyze false positive rate vs. detection rate
- Report optimal operating points under different cost assumptions

**Expected evidence:** PR curves + threshold analysis table + cost-sensitive analysis

---

### RQ6: Cross-Engine Generalization
> How robust are the models when evaluated on machines/operating conditions not represented in training?

**Experimental approach:**
- Engine-level train/val/test splits
- Train on subset of engines, test on held-out engines
- Potentially cross-dataset evaluation (train FD001, test FD002)

**Expected evidence:** Generalization performance table + comparison with in-distribution performance

---

### RQ7: Complexity vs. Improvement
> Does increasing model complexity actually provide meaningful improvements over simpler anomaly-detection approaches?

**Experimental approach:**
- Compare: Statistical → Isolation Forest → Autoencoder → LSTM-AE
- Report computational cost alongside performance
- Analyze marginal improvement per unit of complexity

**Expected evidence:** Performance vs. complexity chart + training/inference time comparison + parameter count

---

## Research Question Mapping to Experiments

| RQ | Primary Experiment | Dataset(s) | Key Metrics |
|----|-------------------|------------|-------------|
| RQ1 | Model comparison | FD001 | F1, PR-AUC, Lead Time |
| RQ2 | Temporal ablation | FD001 | F1, Anomaly trajectories |
| RQ3 | Cross-condition | FD001, FD002, FD004 | Per-condition F1 |
| RQ4 | Early warning | FD001 | Lead time, Detection rate |
| RQ5 | Threshold analysis | FD001 | PR curve, FAR |
| RQ6 | Generalization | FD001, FD002 | Generalization gap |
| RQ7 | Complexity analysis | FD001 | F1/cost ratio |
