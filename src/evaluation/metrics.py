"""Unified evaluation metrics for anomaly detection."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix,
    roc_auc_score
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None
) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'false_alarm_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'miss_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
    }
    if y_scores is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            p, r, _ = precision_recall_curve(y_true, y_scores)
            metrics['pr_auc'] = auc(r, p)
        except ValueError:
            metrics['roc_auc'] = float('nan')
            metrics['pr_auc'] = float('nan')
    return metrics


def create_binary_labels_from_rul(
    rul_values: np.ndarray,
    anomaly_threshold_rul: int = 30
) -> np.ndarray:
    return (rul_values <= anomaly_threshold_rul).astype(int)


def compute_early_warning_metrics(
    engine_ids: np.ndarray,
    cycles: np.ndarray,
    predictions: np.ndarray,
    rul_values: np.ndarray,
    persistence_window: int = 5
) -> Dict[str, any]:
    unique_engines = np.unique(engine_ids)
    lead_times = []
    detected = []
    missed = []
    per_engine = {}
    
    for eid in unique_engines:
        mask = engine_ids == eid
        e_cycles = cycles[mask]
        e_preds = predictions[mask]
        e_ruls = rul_values[mask]
        
        sort_idx = np.argsort(e_cycles)
        e_cycles = e_cycles[sort_idx]
        e_preds = e_preds[sort_idx]
        e_ruls = e_ruls[sort_idx]
        
        max_rul = e_ruls.max()
        first_idx = _find_sustained(e_preds, persistence_window)
        
        if first_idx is not None:
            det_rul = float(e_ruls[first_idx])
            lead_times.append(det_rul)
            detected.append(eid)
            per_engine[int(eid)] = {
                'detected': True,
                'detection_cycle': int(e_cycles[first_idx]),
                'detection_rul': det_rul,
                'lead_time_cycles': det_rul,
                'total_observed_rul': float(max_rul),
            }
        else:
            missed.append(eid)
            per_engine[int(eid)] = {
                'detected': False,
                'detection_cycle': None,
                'detection_rul': None,
                'lead_time_cycles': 0,
                'total_observed_rul': float(max_rul),
            }
    
    n_tot = len(unique_engines)
    n_det = len(detected)
    return {
        'n_engines_total': n_tot,
        'n_engines_detected': n_det,
        'n_engines_missed': len(missed),
        'detection_rate': n_det / n_tot if n_tot > 0 else 0,
        'lead_times': lead_times,
        'mean_lead_time': float(np.mean(lead_times)) if lead_times else 0,
        'median_lead_time': float(np.median(lead_times)) if lead_times else 0,
        'std_lead_time': float(np.std(lead_times)) if lead_times else 0,
        'missed_engine_ids': [int(e) for e in missed],
        'per_engine': per_engine,
        'persistence_window': persistence_window,
    }


def _find_sustained(preds: np.ndarray, window: int) -> Optional[int]:
    if window <= 0:
        window = 1
    count = 0
    for i, p in enumerate(preds):
        if p == 1:
            count += 1
            if count >= window:
                return i - window + 1
        else:
            count = 0
    return None
