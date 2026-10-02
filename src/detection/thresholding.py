"""Anomaly detection threshold selection strategies.

All thresholds are determined from VALIDATION data only.
Test data is NEVER used for threshold selection.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def threshold_percentile(val_scores: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(val_scores, percentile))


def threshold_mean_std(val_scores: np.ndarray, n_std: float = 3.0) -> float:
    return float(np.mean(val_scores) + n_std * np.std(val_scores))


def threshold_iqr(val_scores: np.ndarray, k: float = 1.5) -> float:
    q75 = np.percentile(val_scores, 75)
    q25 = np.percentile(val_scores, 25)
    iqr = q75 - q25
    return float(q75 + k * iqr)


def threshold_best_f1(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    n_candidates: int = 100
) -> Tuple[float, float]:
    from sklearn.metrics import f1_score
    
    candidates = np.linspace(np.min(val_scores), np.max(val_scores), n_candidates)
    best_f1 = 0.0
    best_thresh = candidates[0]
    
    for t in candidates:
        preds = (val_scores >= t).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    
    return float(best_thresh), float(best_f1)


def compute_all_thresholds(
    val_scores: np.ndarray,
    val_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    thresholds = {
        'percentile_90': threshold_percentile(val_scores, 90.0),
        'percentile_95': threshold_percentile(val_scores, 95.0),
        'percentile_99': threshold_percentile(val_scores, 99.0),
        'mean_2std': threshold_mean_std(val_scores, 2.0),
        'mean_3std': threshold_mean_std(val_scores, 3.0),
        'iqr_1.5': threshold_iqr(val_scores, 1.5),
        'iqr_3.0': threshold_iqr(val_scores, 3.0),
    }
    if val_labels is not None:
        thresh, _ = threshold_best_f1(val_scores, val_labels)
        thresholds['best_f1_val'] = thresh
    
    return thresholds
