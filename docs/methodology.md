# Methodology

> **Status:** Draft outline. To be refined as experiments are designed.

## Overview

This project implements a systematic comparison of unsupervised anomaly detection methods for predictive maintenance in Industrial IoT, evaluated under realistic conditions including operating-condition variation, cross-engine generalization, and early-warning capability.

---

## 1. Dataset

### NASA C-MAPSS

- Commercial Modular Aero-Propulsion System Simulation
- Turbofan engine degradation data (run-to-failure)
- 21 sensor measurements + 3 operational settings per cycle
- Multiple subsets with varying complexity

| Subset | Op. Conditions | Fault Modes | Complexity |
|--------|---------------|-------------|------------|
| FD001  | 1             | 1           | Low        |
| FD002  | 6             | 1           | Medium     |
| FD003  | 1             | 2           | Medium     |
| FD004  | 6             | 2           | High       |

---

## 2. Data Pipeline

1. **Load** raw text files
2. **Inspect** feature statistics, constants, correlations
3. **Remove** constant/near-constant features (documented)
4. **Split** at engine level (train / validation / test)
5. **Normalize** using training statistics only
6. **Define** normal operating period (early engine life)
7. **Generate** sequences for temporal models
8. **Verify** no leakage at each step

---

## 3. Models

### 3.1 Statistical Threshold Baseline
- Per-sensor z-score or percentile-based anomaly scoring
- No model training required
- Reference point for all other methods

### 3.2 Isolation Forest
- Ensemble of isolation trees
- Anomaly score from path length
- Trained on normal operating data

### 3.3 Fully Connected Autoencoder
- Encoder → Bottleneck → Decoder
- Trained to reconstruct normal sensor readings
- Anomaly score = reconstruction error (MSE/MAE)

### 3.4 LSTM Autoencoder
- LSTM Encoder → Latent → LSTM Decoder
- Trained on sequences from normal operating periods
- Anomaly score = sequence reconstruction error
- Captures temporal degradation patterns

---

## 4. Anomaly Scoring and Thresholding

*Detailed in `docs/experiment_protocol.md`*

---

## 5. Early Warning Framework

*Detailed in `docs/experiment_protocol.md`*

---

## 6. Evaluation

### Metrics
- Precision, Recall, F1, PR-AUC, ROC-AUC
- False Positive Rate, Detection Rate
- Detection Lead Time, Mean Detection Delay
- Missed Failure Rate
- Training Time, Inference Time, Parameter Count

### Statistical Rigor
- 5 random seeds per experiment
- Mean ± standard deviation reporting
- Paired statistical tests for model comparisons

---

## 7. Experimental Dimensions

1. Model comparison (RQ1, RQ2, RQ7)
2. Cross-condition evaluation (RQ3)
3. Early warning analysis (RQ4)
4. Threshold sensitivity (RQ5)
5. Cross-engine generalization (RQ6)
6. Ablation studies (sequence length, latent dim, sensor subsets)
