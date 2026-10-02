"""Autoencoder training and evaluation runner.

Trains FC-Autoencoder on C-MAPSS FD001 and compares directly against baselines.
Uses identical splits, normalization, thresholds, and metrics.
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

from src.preprocessing.pipeline import DataPipeline
from src.models.autoencoder import AutoencoderDetector
from src.models.baselines import StatisticalThresholdDetector, IsolationForestDetector
from src.detection.thresholding import compute_all_thresholds
from src.evaluation.metrics import (
    compute_classification_metrics, create_binary_labels_from_rul,
    compute_early_warning_metrics
)
from src.utils.reproducibility import set_seed, get_device


def run_experiment():
    print("\n" + "=" * 60)
    print("PHASE 6: AUTOENCODER EXPERIMENTS — C-MAPSS FD001")
    print("=" * 60)
    
    # Load config
    config_path = Path('configs/autoencoder.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)
    device = get_device()
    print(f"\nSeed: {seed} | Device: {device}")
    
    # Data pipeline
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
    input_dim = len(feature_cols)
    
    # Data arrays
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
    
    # Also extract normal data from validation for early stopping
    val_normal_df = data['val_df'].groupby('unit_id').apply(
        lambda g: g.sort_values('cycle').iloc[:max(1, int(len(g) * 0.7))]
    ).reset_index(drop=True)
    val_normal_X = val_normal_df[feature_cols].values
    
    print(f"\nTraining normal samples: {len(train_normal_X)}")
    print(f"Validation normal samples: {len(val_normal_X)}")
    print(f"Input features: {input_dim}")
    
    arch = config.get('model', {}).get('architecture', {})
    train_cfg = config.get('training', {})
    
    # --- Train Autoencoder ---
    print("\n" + "-" * 50)
    print("Training Autoencoder...")
    print("-" * 50)
    
    detector = AutoencoderDetector(
        input_dim=input_dim,
        encoder_dims=arch.get('encoder_layers', [64, 32]),
        latent_dim=arch.get('latent_dim', 16),
        dropout=arch.get('dropout', 0.1),
        activation=arch.get('activation', 'relu'),
        learning_rate=train_cfg.get('learning_rate', 0.001),
        weight_decay=train_cfg.get('weight_decay', 1e-5),
        batch_size=train_cfg.get('batch_size', 256),
        max_epochs=train_cfg.get('epochs', 100),
        patience=train_cfg.get('early_stopping_patience', 10),
        device=device,
        seed=seed
    )
    
    detector.fit(train_normal_X, val_normal_X)
    
    print(f"  Training time: {detector.fit_time_:.2f}s")
    print(f"  Best epoch: {detector.best_epoch_}")
    print(f"  Parameters: {detector.n_parameters_}")
    
    # --- Score ---
    val_scores = detector.score(val_X)
    test_scores = detector.score(test_X)
    
    anomaly_rul = config.get('evaluation', {}).get('anomaly_rul_threshold', 30)
    persistence = config.get('evaluation', {}).get('persistence_window', 5)
    
    # --- Thresholds (from validation only) ---
    val_labels = create_binary_labels_from_rul(val_ruls, anomaly_rul)
    thresholds = compute_all_thresholds(val_scores, val_labels)
    
    print(f"\n  Thresholds computed from validation data:")
    for name, val in sorted(thresholds.items()):
        print(f"    {name}: {val:.6f}")
    
    # --- Evaluate on test data ---
    test_labels = create_binary_labels_from_rul(test_ruls, anomaly_rul)
    
    # Primary threshold
    primary_thresh_name = 'percentile_95'
    primary_thresh = thresholds[primary_thresh_name]
    test_preds = (test_scores >= primary_thresh).astype(int)
    
    cls_metrics = compute_classification_metrics(test_labels, test_preds, test_scores)
    ew_metrics = compute_early_warning_metrics(
        test_engine_ids, test_cycles, test_preds, test_ruls, persistence
    )
    
    print(f"\n  Test Results (threshold: {primary_thresh_name}):")
    print(f"    Precision: {cls_metrics['precision']:.4f}")
    print(f"    Recall:    {cls_metrics['recall']:.4f}")
    print(f"    F1:        {cls_metrics['f1']:.4f}")
    print(f"    FAR:       {cls_metrics['false_alarm_rate']:.4f}")
    print(f"    ROC-AUC:   {cls_metrics['roc_auc']:.4f}")
    print(f"    PR-AUC:    {cls_metrics['pr_auc']:.4f}")
    print(f"    Detection: {ew_metrics['detection_rate']:.2%} ({ew_metrics['n_engines_detected']}/{ew_metrics['n_engines_total']})")
    print(f"    Mean Lead: {ew_metrics['mean_lead_time']:.1f} cycles")
    
    # Threshold sensitivity
    all_threshold_results = {}
    for t_name, t_val in thresholds.items():
        p = (test_scores >= t_val).astype(int)
        m = compute_classification_metrics(test_labels, p, test_scores)
        ew = compute_early_warning_metrics(
            test_engine_ids, test_cycles, p, test_ruls, persistence
        )
        all_threshold_results[t_name] = {
            **m,
            'threshold_value': t_val,
            'detection_rate': ew['detection_rate'],
            'mean_lead_time': ew['mean_lead_time'],
        }
    
    # --- Load baseline results for direct comparison ---
    baseline_path = Path('results/tables/baselines/FD001_baseline_comparison.csv')
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
    else:
        baseline_df = pd.DataFrame()
    
    # Add Autoencoder to comparison
    ae_row = {
        'Model': 'Autoencoder (FC)',
        'Precision': f"{cls_metrics['precision']:.3f}",
        'Recall': f"{cls_metrics['recall']:.3f}",
        'F1': f"{cls_metrics['f1']:.3f}",
        'FAR': f"{cls_metrics['false_alarm_rate']:.3f}",
        'ROC-AUC': f"{cls_metrics['roc_auc']:.3f}",
        'PR-AUC': f"{cls_metrics['pr_auc']:.3f}",
        'Det.Rate': f"{ew_metrics['detection_rate']:.2%}",
        'MeanLead': f"{ew_metrics['mean_lead_time']:.1f}",
        'FitTime': f"{detector.fit_time_:.2f}s",
    }
    
    print("\n" + "=" * 60)
    print("UPDATED COMPARISON TABLE — FD001 (percentile_95)")
    print("=" * 60)
    
    full_comparison = pd.concat([
        baseline_df,
        pd.DataFrame([ae_row])
    ], ignore_index=True)
    
    print(full_comparison.to_string(index=False))
    
    # Save updated comparison
    out_dir = Path('results/tables/comparison')
    out_dir.mkdir(parents=True, exist_ok=True)
    full_comparison.to_csv(out_dir / 'FD001_baselines_vs_autoencoder.csv', index=False)
    
    # Save full AE results
    results_dir = Path('results/logs')
    ae_full = {
        'model_name': 'Autoencoder (FC)',
        'model_params': detector.get_params(),
        'primary_threshold': primary_thresh_name,
        'primary_threshold_value': primary_thresh,
        'classification_metrics': cls_metrics,
        'early_warning': {k: v for k, v in ew_metrics.items() if k != 'per_engine'},
        'per_engine_detection': ew_metrics['per_engine'],
        'threshold_sensitivity': all_threshold_results,
        'all_thresholds': thresholds,
        'train_losses': detector.train_losses,
        'val_losses': detector.val_losses,
    }
    
    with open(results_dir / 'FD001_autoencoder_results.json', 'w') as f:
        json.dump(ae_full, f, indent=2, default=str)
    
    # Save threshold sensitivity table
    thresh_rows = []
    for t_name, t_res in all_threshold_results.items():
        thresh_rows.append({'threshold_strategy': t_name, **t_res})
    pd.DataFrame(thresh_rows).to_csv(
        out_dir / 'FD001_autoencoder_threshold_sensitivity.csv', index=False
    )
    
    print(f"\nResults saved to:")
    print(f"  {out_dir / 'FD001_baselines_vs_autoencoder.csv'}")
    print(f"  {results_dir / 'FD001_autoencoder_results.json'}")
    
    # Print threshold sensitivity
    print("\n" + "=" * 60)
    print("AUTOENCODER THRESHOLD SENSITIVITY")
    print("=" * 60)
    t_display = []
    for t_name, t_res in all_threshold_results.items():
        t_display.append({
            'Threshold': t_name,
            'Value': f"{t_res['threshold_value']:.6f}",
            'Prec': f"{t_res['precision']:.3f}",
            'Rec': f"{t_res['recall']:.3f}",
            'F1': f"{t_res['f1']:.3f}",
            'FAR': f"{t_res['false_alarm_rate']:.3f}",
            'DetRate': f"{t_res['detection_rate']:.2%}",
            'LeadTime': f"{t_res['mean_lead_time']:.1f}",
        })
    print(pd.DataFrame(t_display).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("PHASE 6 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    run_experiment()
