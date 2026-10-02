"""Baseline model training and evaluation runner.

Runs all baseline models (Statistical, IF, OC-SVM) on C-MAPSS FD001
with consistent pipeline, thresholding, and evaluation.

Usage:
    python scripts/run_baselines.py [--config configs/baseline.yaml]
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cmapss_loader import load_train_data
from src.preprocessing.pipeline import DataPipeline
from src.preprocessing.normalization import get_feature_columns
from src.models.baselines import (
    StatisticalThresholdDetector, IsolationForestDetector,
    OneClassSVMDetector, create_detector
)
from src.detection.thresholding import compute_all_thresholds
from src.evaluation.metrics import (
    compute_classification_metrics, create_binary_labels_from_rul,
    compute_early_warning_metrics, format_metrics_table
)
from src.utils.reproducibility import set_seed


def run_single_baseline(
    model_name: str,
    detector,
    train_normal_data: np.ndarray,
    val_data: np.ndarray,
    val_ruls: np.ndarray,
    val_engine_ids: np.ndarray,
    val_cycles: np.ndarray,
    test_data: np.ndarray,
    test_ruls: np.ndarray,
    test_engine_ids: np.ndarray,
    test_cycles: np.ndarray,
    anomaly_rul_threshold: int = 30,
    persistence_window: int = 5,
) -> Dict[str, Any]:
    """Train and evaluate a single baseline model.
    
    Args:
        model_name: Name for logging.
        detector: Detector instance with fit/score interface.
        train_normal_data: Normal training features.
        val_data: Validation features.
        val_ruls: Validation RUL values.
        test_data: Test features.
        test_ruls: Test RUL values.
        anomaly_rul_threshold: RUL below which timestep is anomalous.
        persistence_window: Consecutive anomaly predictions for early warning.
        
    Returns:
        Dictionary with all results.
    """
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    
    # --- TRAIN ---
    print(f"  Training on {len(train_normal_data)} normal samples...")
    fit_start = time.time()
    detector.fit(train_normal_data)
    fit_time = time.time() - fit_start
    print(f"  Training time: {fit_time:.2f}s")
    
    # --- SCORE ---
    print(f"  Scoring validation ({len(val_data)} samples)...")
    score_start = time.time()
    val_scores = detector.score(val_data)
    val_score_time = time.time() - score_start
    
    print(f"  Scoring test ({len(test_data)} samples)...")
    test_scores = detector.score(test_data)
    
    # --- THRESHOLD (from validation only) ---
    val_labels = create_binary_labels_from_rul(val_ruls, anomaly_rul_threshold)
    thresholds = compute_all_thresholds(val_scores, val_labels)
    
    print(f"  Thresholds computed from validation data:")
    for name, val in sorted(thresholds.items()):
        print(f"    {name}: {val:.4f}")
    
    # --- EVALUATE (on test data) ---
    test_labels = create_binary_labels_from_rul(test_ruls, anomaly_rul_threshold)
    
    # Use percentile_95 as primary threshold
    primary_threshold_name = 'percentile_95'
    primary_threshold = thresholds[primary_threshold_name]
    
    test_predictions = (test_scores >= primary_threshold).astype(int)
    
    # Classification metrics
    cls_metrics = compute_classification_metrics(
        test_labels, test_predictions, test_scores
    )
    print(f"\n  Test Results (threshold: {primary_threshold_name}):")
    print(f"    Precision: {cls_metrics['precision']:.4f}")
    print(f"    Recall:    {cls_metrics['recall']:.4f}")
    print(f"    F1:        {cls_metrics['f1']:.4f}")
    print(f"    FAR:       {cls_metrics['false_alarm_rate']:.4f}")
    if 'roc_auc' in cls_metrics:
        print(f"    ROC-AUC:   {cls_metrics['roc_auc']:.4f}")
        print(f"    PR-AUC:    {cls_metrics['pr_auc']:.4f}")
    
    # Early warning metrics
    ew_metrics = compute_early_warning_metrics(
        test_engine_ids, test_cycles, test_predictions,
        test_ruls, persistence_window
    )
    print(f"\n  Early Warning (persistence={persistence_window}):")
    print(f"    Engines detected: {ew_metrics['n_engines_detected']}/{ew_metrics['n_engines_total']}")
    print(f"    Detection rate:   {ew_metrics['detection_rate']:.2%}")
    print(f"    Mean lead time:   {ew_metrics['mean_lead_time']:.1f} cycles")
    print(f"    Median lead time: {ew_metrics['median_lead_time']:.1f} cycles")
    
    # Evaluate across ALL thresholds for sensitivity analysis
    all_threshold_results = {}
    for thresh_name, thresh_val in thresholds.items():
        preds = (test_scores >= thresh_val).astype(int)
        t_metrics = compute_classification_metrics(test_labels, preds, test_scores)
        t_ew = compute_early_warning_metrics(
            test_engine_ids, test_cycles, preds,
            test_ruls, persistence_window
        )
        all_threshold_results[thresh_name] = {
            **t_metrics,
            'threshold_value': thresh_val,
            'detection_rate': t_ew['detection_rate'],
            'mean_lead_time': t_ew['mean_lead_time'],
        }
    
    # Inference timing
    score_time_per_sample = val_score_time / len(val_data) * 1000  # ms
    
    result = {
        'model_name': model_name,
        'model_params': detector.get_params(),
        'primary_threshold': primary_threshold_name,
        'primary_threshold_value': primary_threshold,
        'classification_metrics': cls_metrics,
        'early_warning': {
            k: v for k, v in ew_metrics.items() if k != 'per_engine'
        },
        'per_engine_detection': ew_metrics['per_engine'],
        'all_thresholds': thresholds,
        'threshold_sensitivity': all_threshold_results,
        'timing': {
            'fit_seconds': fit_time,
            'score_ms_per_sample': score_time_per_sample,
        },
        'anomaly_rul_threshold': anomaly_rul_threshold,
    }
    
    return result


def main():
    print("\n" + "=" * 60)
    print("BASELINE EXPERIMENTS — C-MAPSS FD001")
    print("=" * 60)
    
    # Load config
    config_path = Path('configs/baseline.yaml')
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Seed for reproducibility
    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)
    print(f"\nSeed: {seed}")
    
    # Prepare data pipeline
    pipeline_config = {
        'experiment': {'seed': seed},
        'data': {
            'dataset': 'FD001',
            'data_dir': 'data/raw',
            'output_dir': 'data/processed',
            'train_ratio': 0.70,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'normalization': 'standard',
        },
        'features': {
            'selected_sensors': None,
            'include_operational_settings': True,
        },
        'training': {
            'normal_ratio': 0.7,
        },
    }
    
    pipeline = DataPipeline(pipeline_config)
    data = pipeline.prepare(verbose=True)
    
    feature_cols = data['feature_cols']
    
    # Prepare feature arrays
    train_normal_X = data['train_normal_df'][feature_cols].values
    
    val_df = data['val_df']
    val_X = val_df[feature_cols].values
    val_ruls = val_df['RUL'].values
    val_engine_ids = val_df['unit_id'].values
    val_cycles = val_df['cycle'].values
    
    test_df = data['test_df']
    test_X = test_df[feature_cols].values
    test_ruls = test_df['RUL'].values
    test_engine_ids = test_df['unit_id'].values
    test_cycles = test_df['cycle'].values
    
    print(f"\nTrain normal: {train_normal_X.shape}")
    print(f"Validation:   {val_X.shape}")
    print(f"Test:         {test_X.shape}")
    
    anomaly_rul = config.get('evaluation', {}).get('anomaly_rul_threshold', 30)
    persistence = config.get('evaluation', {}).get('persistence_window', 5)
    
    # --- Run baselines ---
    all_results = {}
    
    # 1. Statistical Threshold (mean z-score)
    stat_detector = StatisticalThresholdDetector(aggregation='mean')
    all_results['Statistical_mean'] = run_single_baseline(
        'Statistical (mean z-score)', stat_detector,
        train_normal_X, val_X, val_ruls, val_engine_ids, val_cycles,
        test_X, test_ruls, test_engine_ids, test_cycles,
        anomaly_rul, persistence
    )
    
    # 2. Statistical Threshold (max z-score)
    stat_max = StatisticalThresholdDetector(aggregation='max')
    all_results['Statistical_max'] = run_single_baseline(
        'Statistical (max z-score)', stat_max,
        train_normal_X, val_X, val_ruls, val_engine_ids, val_cycles,
        test_X, test_ruls, test_engine_ids, test_cycles,
        anomaly_rul, persistence
    )
    
    # 3. Isolation Forest
    if_detector = IsolationForestDetector(
        n_estimators=config.get('isolation_forest', {}).get('n_estimators', 200),
        random_state=seed
    )
    all_results['IsolationForest'] = run_single_baseline(
        'Isolation Forest', if_detector,
        train_normal_X, val_X, val_ruls, val_engine_ids, val_cycles,
        test_X, test_ruls, test_engine_ids, test_cycles,
        anomaly_rul, persistence
    )
    
    # 4. One-Class SVM
    ocsvm_detector = OneClassSVMDetector(
        kernel='rbf',
        nu=config.get('ocsvm', {}).get('nu', 0.1),
        max_train_samples=config.get('ocsvm', {}).get('max_train_samples', 5000)
    )
    all_results['OneClassSVM'] = run_single_baseline(
        'One-Class SVM', ocsvm_detector,
        train_normal_X, val_X, val_ruls, val_engine_ids, val_cycles,
        test_X, test_ruls, test_engine_ids, test_cycles,
        anomaly_rul, persistence
    )
    
    # --- Summary table ---
    print("\n" + "=" * 60)
    print("COMPARISON TABLE — Primary Threshold (percentile_95)")
    print("=" * 60)
    
    summary_rows = []
    for name, result in all_results.items():
        cm = result['classification_metrics']
        ew = result['early_warning']
        tm = result['timing']
        summary_rows.append({
            'Model': name,
            'Precision': f"{cm['precision']:.3f}",
            'Recall': f"{cm['recall']:.3f}",
            'F1': f"{cm['f1']:.3f}",
            'FAR': f"{cm['false_alarm_rate']:.3f}",
            'ROC-AUC': f"{cm.get('roc_auc', float('nan')):.3f}",
            'PR-AUC': f"{cm.get('pr_auc', float('nan')):.3f}",
            'Det.Rate': f"{ew['detection_rate']:.2%}",
            'MeanLead': f"{ew['mean_lead_time']:.1f}",
            'FitTime': f"{tm['fit_seconds']:.2f}s",
        })
    
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    # --- Save results ---
    output_dir = Path('results/tables/baselines')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_df.to_csv(output_dir / 'FD001_baseline_comparison.csv', index=False)
    
    # Save full results JSON
    results_dir = Path('results/logs')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove non-serializable items
    serializable = {}
    for name, result in all_results.items():
        r = {k: v for k, v in result.items()}
        serializable[name] = r
    
    with open(results_dir / 'FD001_baseline_results.json', 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    
    print(f"\nResults saved to:")
    print(f"  {output_dir / 'FD001_baseline_comparison.csv'}")
    print(f"  {results_dir / 'FD001_baseline_results.json'}")
    
    # --- Threshold sensitivity table ---
    print("\n" + "=" * 60)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    for model_name, result in all_results.items():
        print(f"\n{model_name}:")
        rows = []
        for thresh_name, thresh_result in result['threshold_sensitivity'].items():
            rows.append({
                'Threshold': thresh_name,
                'Value': f"{thresh_result['threshold_value']:.4f}",
                'Prec': f"{thresh_result['precision']:.3f}",
                'Rec': f"{thresh_result['recall']:.3f}",
                'F1': f"{thresh_result['f1']:.3f}",
                'FAR': f"{thresh_result['false_alarm_rate']:.3f}",
                'DetRate': f"{thresh_result['detection_rate']:.2%}",
                'LeadTime': f"{thresh_result['mean_lead_time']:.1f}",
            })
        thresh_df = pd.DataFrame(rows)
        print(thresh_df.to_string(index=False))
    
    # Save threshold sensitivity
    for model_name, result in all_results.items():
        rows = []
        for thresh_name, thresh_result in result['threshold_sensitivity'].items():
            row = {'threshold_strategy': thresh_name, **thresh_result}
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            output_dir / f'FD001_{model_name}_threshold_sensitivity.csv',
            index=False
        )
    
    print("\n" + "=" * 60)
    print("BASELINE EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
