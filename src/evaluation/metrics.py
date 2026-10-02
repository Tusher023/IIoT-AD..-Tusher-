"""Unified evaluation metrics for anomaly detection.

Provides classification metrics, early-warning metrics, and
per-engine analysis — all computed consistently across methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
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
    """Compute standard classification metrics.
    
    Args:
        y_true: Ground truth binary labels (1=anomaly, 0=normal).
        y_pred: Predicted binary labels.
        y_scores: Continuous anomaly scores (for AUC).
        
    Returns:
        Dictionary of metric name -> value.
    """
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
            prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_scores)
            metrics['pr_auc'] = auc(rec_vals, prec_vals)
        except ValueError:
            metrics['roc_auc'] = float('nan')
            metrics['pr_auc'] = float('nan')
    
    return metrics


def create_binary_labels_from_rul(
    rul_values: np.ndarray,
    anomaly_threshold_rul: int = 30
) -> np.ndarray:
    """Create binary anomaly labels from RUL values.
    
    A timestep is labeled as "anomalous" (degraded) if the RUL
    is below the threshold, meaning the engine is close to failure.
    
    Args:
        rul_values: Remaining Useful Life values.
        anomaly_threshold_rul: RUL below which a timestep is anomalous.
        
    Returns:
        Binary labels (1=anomaly/degraded, 0=normal).
    """
    return (rul_values <= anomaly_threshold_rul).astype(int)


def compute_early_warning_metrics(
    engine_ids: np.ndarray,
    cycles: np.ndarray,
    predictions: np.ndarray,
    rul_values: np.ndarray,
    persistence_window: int = 5
) -> Dict[str, any]:
    """Compute early-warning / detection lead time metrics.
    
    For each engine, finds the first SUSTAINED anomaly detection
    (where at least `persistence_window` consecutive predictions are 1)
    and computes how many cycles before failure it was triggered.
    
    Args:
        engine_ids: Engine ID for each observation.
        cycles: Cycle number for each observation.
        predictions: Binary predictions (1=anomaly).
        rul_values: RUL at each observation.
        persistence_window: Min consecutive anomaly predictions for a "sustained" alert.
        
    Returns:
        Dictionary with per-engine and aggregate early-warning metrics.
    """
    unique_engines = np.unique(engine_ids)
    
    lead_times = []
    detected_engines = []
    missed_engines = []
    
    per_engine = {}
    
    for eid in unique_engines:
        mask = engine_ids == eid
        eng_cycles = cycles[mask]
        eng_preds = predictions[mask]
        eng_ruls = rul_values[mask]
        
        # Sort by cycle
        sort_idx = np.argsort(eng_cycles)
        eng_cycles = eng_cycles[sort_idx]
        eng_preds = eng_preds[sort_idx]
        eng_ruls = eng_ruls[sort_idx]
        
        max_rul = eng_ruls.max()  # RUL at first cycle ~ total remaining life
        
        # Find first sustained anomaly detection
        first_detection_idx = _find_sustained_detection(eng_preds, persistence_window)
        
        if first_detection_idx is not None:
            detection_rul = eng_ruls[first_detection_idx]
            detection_cycle = eng_cycles[first_detection_idx]
            lead_times.append(float(detection_rul))
            detected_engines.append(eid)
            
            per_engine[int(eid)] = {
                'detected': True,
                'detection_cycle': int(detection_cycle),
                'detection_rul': float(detection_rul),
                'lead_time_cycles': float(detection_rul),
                'total_observed_rul': float(max_rul),
                'detection_fraction': float(detection_rul / max_rul) if max_rul > 0 else 0,
            }
        else:
            missed_engines.append(eid)
            per_engine[int(eid)] = {
                'detected': False,
                'detection_cycle': None,
                'detection_rul': None,
                'lead_time_cycles': 0,
                'total_observed_rul': float(max_rul),
                'detection_fraction': 0,
            }
    
    n_total = len(unique_engines)
    n_detected = len(detected_engines)
    
    result = {
        'n_engines_total': n_total,
        'n_engines_detected': n_detected,
        'n_engines_missed': len(missed_engines),
        'detection_rate': n_detected / n_total if n_total > 0 else 0,
        'lead_times': lead_times,
        'mean_lead_time': float(np.mean(lead_times)) if lead_times else 0,
        'median_lead_time': float(np.median(lead_times)) if lead_times else 0,
        'std_lead_time': float(np.std(lead_times)) if lead_times else 0,
        'min_lead_time': float(np.min(lead_times)) if lead_times else 0,
        'max_lead_time': float(np.max(lead_times)) if lead_times else 0,
        'missed_engine_ids': [int(e) for e in missed_engines],
        'per_engine': per_engine,
        'persistence_window': persistence_window,
    }
    
    return result


def _find_sustained_detection(
    predictions: np.ndarray,
    persistence_window: int
) -> Optional[int]:
    """Find the index of the first sustained anomaly detection.
    
    Returns the index where `persistence_window` consecutive 1s START.
    """
    if persistence_window <= 0:
        persistence_window = 1
    
    count = 0
    for i, pred in enumerate(predictions):
        if pred == 1:
            count += 1
            if count >= persistence_window:
                return i - persistence_window + 1
        else:
            count = 0
    
    return None


def format_metrics_table(
    all_results: Dict[str, Dict],
    metric_keys: Optional[List[str]] = None
) -> pd.DataFrame:
    """Format results from multiple models into a comparison table.
    
    Args:
        all_results: {model_name: {metric: value}} dictionary.
        metric_keys: Specific metrics to include (None = all).
        
    Returns:
        DataFrame with models as rows and metrics as columns.
    """
    rows = []
    for model_name, metrics in all_results.items():
        row = {'model': model_name}
        if metric_keys:
            for key in metric_keys:
                row[key] = metrics.get(key, float('nan'))
        else:
            row.update(metrics)
        rows.append(row)
    
    return pd.DataFrame(rows).set_index('model')
