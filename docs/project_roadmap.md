# Project Roadmap

## Working Title
"Unsupervised Anomaly Detection for Predictive Maintenance in Industrial IoT Using Autoencoder and LSTM-Based Models"

---

## Research Timeline

| Phase | Description | Status | Dependencies |
|-------|-------------|--------|--------------|
| Phase 0 | Project Foundation & Environment Setup | ✅ Complete | None |
| Phase 1 | Literature Review & Research Landscape | 🔲 Pending | Phase 0 |
| Phase 2 | Research Gap Definition | 🔲 Pending | Phase 1 |
| Phase 3 | Dataset Acquisition & EDA | 🔲 Pending | Phase 0 |
| Phase 4 | Data Engineering Pipeline | 🔲 Pending | Phase 3 |
| Phase 5 | Baseline Implementation | 🔲 Pending | Phase 4 |
| Phase 6 | Autoencoder Implementation | 🔲 Pending | Phase 4 |
| Phase 7 | LSTM Autoencoder Implementation | 🔲 Pending | Phase 4 |
| Phase 8 | Anomaly Scoring & Thresholding | 🔲 Pending | Phases 5-7 |
| Phase 9 | Early Warning Framework | 🔲 Pending | Phase 8 |
| Phase 10 | Cross-Condition Experiments | 🔲 Pending | Phase 8 |
| Phase 11 | Ablation Studies | 🔲 Pending | Phase 8 |
| Phase 12 | Statistical Analysis | 🔲 Pending | Phases 10-11 |
| Phase 13 | Publication-Quality Figures & Tables | 🔲 Pending | Phase 12 |
| Phase 14 | Demonstration Dashboard | 🔲 Pending | Phase 8 |
| Phase 15 | Research Paper Draft | 🔲 Pending | Phase 13 |
| Phase 16 | BTech Thesis Draft | 🔲 Pending | Phase 15 |
| Phase 17 | Reproducibility & Validity Audit | 🔲 Pending | Phase 16 |

---

## Methodology Overview

### Models Under Investigation

1. **Statistical Threshold Baseline** — Simple statistical anomaly detection (z-score / percentile)
2. **Isolation Forest** — Classical unsupervised anomaly detection
3. **One-Class SVM** — Kernel-based anomaly detection (if computationally feasible)
4. **Fully Connected Autoencoder** — Reconstruction-error-based anomaly detection
5. **LSTM Autoencoder** — Temporal reconstruction-error-based anomaly detection
6. **Variational Autoencoder** — Only if it adds genuine research value

### Datasets

- **Primary:** NASA C-MAPSS Turbofan Engine Degradation (FD001–FD004)
- **Secondary:** To be determined based on research needs

### Key Experimental Dimensions

1. **Model comparison** — Baseline vs. AE vs. LSTM-AE
2. **Operating conditions** — Single vs. multiple conditions (FD001 vs FD002/FD004)
3. **Failure modes** — Single vs. multiple fault modes (FD001 vs FD003/FD004)
4. **Cross-engine generalization** — Train/test on separate engines
5. **Early warning capability** — Detection lead time before failure
6. **Threshold sensitivity** — Effect of threshold strategy on detection quality
7. **Complexity vs. performance** — Is added model complexity justified?

---

## Hardware & Environment

- **OS:** Windows 11 (x64)
- **CPU:** Intel Core i7-1165G7 (4 cores, 8 threads)
- **RAM:** ~16 GB
- **GPU:** Intel Iris Xe Graphics (integrated — no CUDA)
- **Disk:** ~106 GB free
- **Python:** 3.10.7
- **Deep Learning:** PyTorch 2.3.0+cpu

### Hardware Implications

- All training will run on **CPU**. This constrains model size and experiment count.
- LSTM Autoencoder training may be slow — keep architectures moderately sized.
- Batch experiments should use reasonable hyperparameter ranges, not exhaustive grid search.
- Save checkpoints to avoid retraining.

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| No GPU — slow training | Medium | Keep models small, use early stopping, cache results |
| C-MAPSS is synthetic data | Medium | Acknowledge in limitations, discuss implications |
| Overfitting on small engine count | Medium | Engine-level splits, report confidence intervals |
| Threshold sensitivity dominates results | High | Systematic threshold analysis, report across thresholds |
| Deep learning underperforms baselines | Low (risk to expectations) | Report honestly — this IS a valid finding |
| Disk space constraints | Low | Clean intermediate files, use compressed formats |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-10-02 | Use CPU-only PyTorch | No NVIDIA GPU available |
| 2026-10-02 | Start with FD001 | Simplest subset (1 operating condition, 1 fault mode) |
| 2026-10-02 | Engine-level data splits | Prevents temporal leakage across engines |
| 2026-10-02 | Configuration-driven experiments | Reproducibility and systematic variation |
