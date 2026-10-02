"""Baseline model training and evaluation runner."""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import DataPipeline
from src.models.baselines import (
    StatisticalThresholdDetector, IsolationForestDetector,
    OneClassSVMDetector
)
from src.detection.thresholding import compute_all_thresholds
from src.evaluation.metrics import (
    compute_classification_metrics, create_binary_labels_from_rul,
    compute_early_warning_metrics
)
from src.utils.reproducibility import set_seed


def run_single_baseline(
    model_name: str, detector, train_normal_data, val_data, val_ruls,
    val_engine_ids, val_cycles, test_data, test_ruls, test_engine_ids,
    test_cycles, anomaly_rul_threshold=30, persistence_window=5
):
    print(f"\nModel: {model_name}")
    start = time.time()
    detector.fit(train_normal_data)
    fit_time = time.time() - start
    
    val_scores = detector.score(val_data)
    test_scores = detector.score(test_data)
    
    val_labels = create_binary_labels_from_rul(val_ruls, anomaly_rul_threshold)
    thresholds = compute_all_thresholds(val_scores, val_labels)
    
    test_labels = create_binary_labels_from_rul(test_ruls, anomaly_rul_threshold)
    primary_thresh = thresholds['percentile_95']
    test_preds = (test_scores >= primary_thresh).astype(int)
    
    cls_metrics = compute_classification_metrics(test_labels, test_preds, test_scores)
    ew_metrics = compute_early_warning_metrics(
        test_engine_ids, test_cycles, test_preds, test_ruls, persistence_window
    )
    
    all_thresh_results = {}
    for t_name, t_val in thresholds.items():
        preds = (test_scores >= t_val).astype(int)
        m = compute_classification_metrics(test_labels, preds, test_scores)
        ew = compute_early_warning_metrics(
            test_engine_ids, test_cycles, preds, test_ruls, persistence_window
        )
        all_thresh_results[t_name] = {
            **m, 'threshold_value': t_val,
            'detection_rate': ew['detection_rate'],
            'mean_lead_time': ew['mean_lead_time'],
        }
    
    return {
        'model_name': model_name,
        'model_params': detector.get_params(),
        'primary_threshold': 'percentile_95',
        'primary_threshold_value': primary_thresh,
        'classification_metrics': cls_metrics,
        'early_warning': {k: v for k, v in ew_metrics.items() if k != 'per_engine'},
        'per_engine_detection': ew_metrics['per_engine'],
        'threshold_sensitivity': all_thresh_results,
        'all_thresholds': thresholds,
        'timing': {'fit_seconds': fit_time},
    }


def main():
    print("Running baselines...")
    set_seed(42)
    
    pipeline = DataPipeline({
        'experiment': {'seed': 42},
        'data': {'dataset': 'FD001', 'data_dir': 'data/raw', 'output_dir': 'data/processed'},
        'features': {'selected_sensors': None, 'include_operational_settings': True},
        'training': {'normal_ratio': 0.7},
    })
    data = pipeline.prepare(verbose=False)
    fc = data['feature_cols']
    
    train_norm = data['train_normal_df'][fc].values
    val_X = data['val_df'][fc].values
    val_r = data['val_df']['RUL'].values
    val_e = data['val_df']['unit_id'].values
    val_c = data['val_df']['cycle'].values
    
    test_X = data['test_df'][fc].values
    test_r = data['test_df']['RUL'].values
    test_e = data['test_df']['unit_id'].values
    test_c = data['test_df']['cycle'].values
    
    results = {}
    results['Statistical_mean'] = run_single_baseline(
        'Statistical (mean z)', StatisticalThresholdDetector('mean'),
        train_norm, val_X, val_r, val_e, val_c, test_X, test_r, test_e, test_c
    )
    results['IsolationForest'] = run_single_baseline(
        'Isolation Forest', IsolationForestDetector(n_estimators=200, random_state=42),
        train_norm, val_X, val_r, val_e, val_c, test_X, test_r, test_e, test_c
    )
    results['OneClassSVM'] = run_single_baseline(
        'One-Class SVM', OneClassSVMDetector(kernel='rbf', nu=0.1, max_train_samples=5000),
        train_norm, val_X, val_r, val_e, val_c, test_X, test_r, test_e, test_c
    )
    
    rows = []
    for name, r in results.items():
        cm = r['classification_metrics']
        ew = r['early_warning']
        rows.append({
            'Model': name,
            'Precision': f"{cm['precision']:.3f}",
            'Recall': f"{cm['recall']:.3f}",
            'F1': f"{cm['f1']:.3f}",
            'FAR': f"{cm['false_alarm_rate']:.3f}",
            'ROC-AUC': f"{cm.get('roc_auc', float('nan')):.3f}",
            'PR-AUC': f"{cm.get('pr_auc', float('nan')):.3f}",
            'Det.Rate': f"{ew['detection_rate']:.2%}",
            'MeanLead': f"{ew['mean_lead_time']:.1f}",
            'FitTime': f"{r['timing']['fit_seconds']:.2f}s",
        })
    
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    
    out_dir = Path('results/tables/baselines')
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'FD001_baseline_comparison.csv', index=False)
    print("Saved baseline comparison table.")


if __name__ == '__main__':
    main()
