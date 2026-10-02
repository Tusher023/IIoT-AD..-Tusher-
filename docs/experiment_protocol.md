# Experiment Protocol

## Purpose
This document defines the experimental methodology, ensuring all experiments are reproducible, fair, and free from data leakage.

---

## 1. Data Splitting Strategy

### Engine-Level Splits
All splits are performed at the **engine level**, not the observation level.

- **Training set (70% of engines):** Used for model training and learning normal behavior patterns.
- **Validation set (15% of engines):** Used for hyperparameter tuning, threshold selection, and early stopping.
- **Test set (15% of engines):** Used ONLY for final evaluation. Never accessed during model development.

### Split Procedure
1. List all unique engine IDs in the dataset.
2. Shuffle engine IDs using a fixed random seed.
3. Assign engines to train/val/test by the specified ratios.
4. Verify no engine appears in multiple splits.
5. Log the split assignment.

### Rationale
Random observation-level splits would create temporal leakage — a model could see late-life (degraded) observations from an engine in training and early-life observations in testing, artificially inflating performance.

---

## 2. Normalization Protocol

### Procedure
1. Compute normalization statistics (mean, std for StandardScaler; min, max for MinMaxScaler) **using training data only**.
2. Apply the same transformation to validation and test data.
3. Save the scaler object for reproducibility.

### Prohibited
- Computing statistics on the full dataset
- Fitting scaler on validation or test data
- Re-fitting scaler for each experiment without documentation

---

## 3. Normal Data Definition

For autoencoder-based models, we train on "normal" operating data.

### Definition
The first `normal_ratio` fraction (default: 70%) of each training engine's life is considered "normal."

### Justification
C-MAPSS engines start healthy and degrade toward failure. Early cycles represent normal operation. The exact boundary is approximate — this is acknowledged as a limitation.

### Alternative
If RUL labels are used to define normal periods, this must be documented and justified. Any use of labels must be restricted to the training set.

---

## 4. Sequence Generation (LSTM Models)

### Sliding Window
- Window size: configurable (default: 30 cycles)
- Stride: configurable (default: 1)
- **No cross-engine sequences**: Windows never span engine boundaries.
- **No future information**: Each window contains only past/current timesteps.

---

## 5. Threshold Selection

### Protocol
1. Train the model on training data.
2. Compute anomaly scores on **validation data**.
3. Select threshold using the validation anomaly score distribution.
4. Apply threshold to **test data** for final evaluation.

### Threshold Strategies Under Investigation
- **Percentile-based:** Threshold = P-th percentile of validation anomaly scores
- **F1-optimized:** Threshold that maximizes F1 on validation set (requires validation labels)
- **Statistical:** Threshold = mean + k × std of training reconstruction errors

### Prohibited
- Selecting threshold by looking at test performance
- Tuning threshold to maximize test metrics
- Changing threshold after observing test results without documentation

---

## 6. Early Warning Framework

### Definitions
- **Anomaly flag:** A single cycle where the anomaly score exceeds the threshold.
- **Warning trigger:** The anomaly score exceeds the threshold for `persistence_window` (default: 5) consecutive cycles.
- **First detection time:** The cycle at which the first warning is triggered.
- **Lead time:** Total engine life − first detection time (cycles before failure).
- **False alarm:** A warning triggered on an engine during its "normal" operating period.
- **Missed failure:** An engine that fails without any prior warning.

### Metrics
- **Detection rate:** Fraction of failed engines where a warning was triggered before failure.
- **Mean lead time:** Average lead time across detected engines.
- **False alarm rate:** Fraction of warning-free normal periods incorrectly flagged.
- **Mean detection delay:** Average cycles from first anomaly to warning trigger.

---

## 7. Evaluation Protocol

### Binary Classification Metrics
For each cycle, determine ground truth: normal or anomalous.

**Ground truth definition:** A cycle is "anomalous" if the engine's remaining useful life (RUL) at that cycle is below a threshold (e.g., RUL < 30 cycles). This threshold should be systematically investigated.

### Metrics Computed
| Metric | Formula | Use |
|--------|---------|-----|
| Precision | TP / (TP + FP) | False alarm cost |
| Recall | TP / (TP + FN) | Missed failure cost |
| F1 | 2 × P × R / (P + R) | Balanced performance |
| PR-AUC | Area under PR curve | Threshold-independent ranking |
| ROC-AUC | Area under ROC curve | Overall discrimination |
| FPR | FP / (FP + TN) | False alarm frequency |

### Early Warning Metrics
| Metric | Description |
|--------|-------------|
| Detection Rate | % of engines with warning before failure |
| Mean Lead Time | Average cycles of advance warning |
| Median Lead Time | Median cycles of advance warning |
| False Alarm Rate | % of normal-phase engines with false warnings |
| Missed Failure Rate | % of failed engines with no warning |

---

## 8. Reproducibility Requirements

### Every Experiment Must Record
- Experiment ID (e.g., `AE_FD001_SEQ30_SEED42`)
- Dataset and subset
- Train/val/test engine IDs
- Feature selection
- Normalization method and parameters
- Model architecture and hyperparameters
- Training configuration (epochs, batch size, LR, etc.)
- Random seed
- Threshold strategy and value
- All evaluation metrics
- Training time and inference time
- Software versions

### Configuration Files
All experiments are driven by YAML configuration files in `configs/`.

### Results Storage
Results are stored as structured JSON in `results/` with the experiment ID.

---

## 9. Statistical Rigor

### Repeated Experiments
For final reported results, run each experiment with **5 different random seeds** (42, 123, 456, 789, 1024).

### Reporting
Report: mean ± standard deviation for all key metrics.

### Statistical Tests
When comparing two models:
- Use paired tests (e.g., Wilcoxon signed-rank test) across seeds/engines.
- Report p-values.
- Do not claim significance without appropriate testing.

---

## 10. Leakage Prevention Checklist

- [ ] Engine-level splits verified
- [ ] Normalization fitted on training data only
- [ ] Threshold selected on validation data only
- [ ] No test data accessed during model development
- [ ] Sequences do not cross engine boundaries
- [ ] Sequences do not use future information
- [ ] Results reported only on held-out test engines
- [ ] No hyperparameter decisions based on test performance
