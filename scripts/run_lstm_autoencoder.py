"""LSTM Autoencoder training and evaluation runner.

Trains LSTM-AE on C-MAPSS FD001 sequences and compares directly against
Baselines and FC-Autoencoder under the unified evaluation framework.
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
from src.preprocessing.sequences import create_sequences, extract_normal_data
from src.models.lstm_autoencoder import LSTMAutoencoderDetector
from src.detection.thresholding import compute_all_thresholds
from src.evaluation.metrics import (
    compute_classification_metrics, create_binary_labels_from_rul,
    compute_early_warning_metrics
)
from src.utils.reproducibility import set_seed, get_device


def run_experiment():
    print("\n" + "=" * 60)
    print("PHASE 7: LSTM AUTOENCODER EXPERIMENTS — C-MAPSS FD001")
    print("=" * 60)

    config_path = Path('configs/lstm_autoencoder.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)
    device = get_device()
    print(f"\nSeed: {seed} | Device: {device}")

    # Data pipeline configuration
    data_cfg = config.get('data', {})
    seq_len = data_cfg.get('sequence_length', 30)
    seq_stride = data_cfg.get('sequence_stride', 1)

    pipeline_config = {
        'experiment': {'seed': seed},
        'data': {
            'dataset': data_cfg.get('dataset', 'FD001'),
            'data_dir': data_cfg.get('data_dir', 'data/raw'),
            'output_dir': data_cfg.get('output_dir', 'data/processed'),
            'train_ratio': data_cfg.get('train_ratio', 0.70),
            'val_ratio': data_cfg.get('val_ratio', 0.15),
            'test_ratio': data_cfg.get('test_ratio', 0.15),
            'normalization': data_cfg.get('normalization', 'standard'),
            'sequence_length': seq_len,
        },
        'features': {
            'selected_sensors': config.get('features', {}).get('selected_sensors', None),
            'include_operational_settings': config.get('features', {}).get('include_operational_settings', True),
        },
        'training': {
            'normal_ratio': config.get('training', {}).get('normal_ratio', 0.7),
        },
    }

    pipeline = DataPipeline(pipeline_config)
    data = pipeline.prepare(verbose=True)

    feature_cols = data['feature_cols']
    input_dim = len(feature_cols)

    # Generate sequence datasets
    seq_data = pipeline.prepare_sequences(data, sequence_length=seq_len, stride=seq_stride)
    train_seqs = seq_data['train_sequences']
    val_seqs = seq_data['val_sequences']
    val_ruls = seq_data['val_ruls']
    val_eids = seq_data['val_engine_ids']
    val_cycles = seq_data['val_cycles']

    test_seqs = seq_data['test_sequences']
    test_ruls = seq_data['test_ruls']
    test_eids = seq_data['test_engine_ids']
    test_cycles = seq_data['test_cycles']

    # Also build normal validation sequences for early stopping
    val_normal_df = extract_normal_data(data['val_df'], normal_ratio=0.7)
    val_normal_seqs, _, _ = create_sequences(
        val_normal_df, feature_cols, sequence_length=seq_len, stride=seq_stride
    )

    print(f"\nTraining normal sequences: {train_seqs.shape}")
    print(f"Validation normal sequences: {val_normal_seqs.shape}")
    print(f"Validation full sequences:   {val_seqs.shape}")
    print(f"Test full sequences:         {test_seqs.shape}")

    # Model configuration
    model_cfg = config.get('model', {})
    enc_cfg = model_cfg.get('encoder', {})
    train_params = config.get('training', {})

    detector = LSTMAutoencoderDetector(
        input_dim=input_dim,
        seq_len=seq_len,
        hidden_dim=enc_cfg.get('hidden_dim', 64),
        latent_dim=model_cfg.get('latent_dim', 32),
        num_layers=enc_cfg.get('num_layers', 2),
        dropout=enc_cfg.get('dropout', 0.2),
        bidirectional=enc_cfg.get('bidirectional', False),
        learning_rate=train_params.get('learning_rate', 0.001),
        weight_decay=train_params.get('weight_decay', 1e-4),
        batch_size=train_params.get('batch_size', 128),
        max_epochs=train_params.get('epochs', 100),
        patience=train_params.get('early_stopping', {}).get('patience', 15),
        clip_grad_norm=train_params.get('clip_grad_norm', 1.0),
        loss_type=train_params.get('loss', 'mse'),
        temporal_aggregate=config.get('anomaly_scoring', {}).get('temporal_aggregate', 'mean'),
        device=device,
        seed=seed
    )

    print("\n" + "-" * 50)
    print("Training LSTM Autoencoder...")
    print("-" * 50)
    detector.fit(train_seqs, val_normal_seqs)

    print(f"  Training time: {detector.fit_time_:.2f}s")
    print(f"  Best epoch: {detector.best_epoch_}")
    print(f"  Parameters: {detector.n_parameters_}")

    # Scoring
    print("\nScoring validation & test sequences...")
    val_scores = detector.score(val_seqs)
    test_scores = detector.score(test_seqs)

    anomaly_rul = 30
    persistence = config.get('evaluation', {}).get('early_warning', {}).get('persistence_window', 5)

    # Thresholds from validation sequences only
    val_labels = create_binary_labels_from_rul(val_ruls, anomaly_rul)
    thresholds = compute_all_thresholds(val_scores, val_labels)

    print(f"\n  Thresholds computed from validation sequences:")
    for name, val in sorted(thresholds.items()):
        print(f"    {name}: {val:.6f}")

    # Evaluate on test sequences
    test_labels = create_binary_labels_from_rul(test_ruls, anomaly_rul)

    primary_thresh_name = 'percentile_95'
    primary_thresh = thresholds[primary_thresh_name]
    test_preds = (test_scores >= primary_thresh).astype(int)

    cls_metrics = compute_classification_metrics(test_labels, test_preds, test_scores)
    ew_metrics = compute_early_warning_metrics(
        test_eids, test_cycles, test_preds, test_ruls, persistence
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

    # Threshold sensitivity across all options
    all_threshold_results = {}
    for t_name, t_val in thresholds.items():
        p = (test_scores >= t_val).astype(int)
        m = compute_classification_metrics(test_labels, p, test_scores)
        ew = compute_early_warning_metrics(
            test_eids, test_cycles, p, test_ruls, persistence
        )
        all_threshold_results[t_name] = {
            **m,
            'threshold_value': t_val,
            'detection_rate': ew['detection_rate'],
            'mean_lead_time': ew['mean_lead_time'],
        }

    # Load existing comparison
    comp_file = Path('results/tables/comparison/FD001_baselines_vs_autoencoder.csv')
    if comp_file.exists():
        prev_df = pd.read_csv(comp_file)
    else:
        prev_df = pd.DataFrame()

    lstm_row = {
        'Model': 'LSTM Autoencoder',
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

    full_comparison = pd.concat([prev_df, pd.DataFrame([lstm_row])], ignore_index=True)

    print("\n" + "=" * 60)
    print("MASTER COMPARISON TABLE — FD001 (percentile_95)")
    print("=" * 60)
    print(full_comparison.to_string(index=False))

    out_dir = Path('results/tables/comparison')
    out_dir.mkdir(parents=True, exist_ok=True)
    full_comparison.to_csv(out_dir / 'FD001_all_models_comparison.csv', index=False)

    # Save detailed logs
    results_dir = Path('results/logs')
    results_dir.mkdir(parents=True, exist_ok=True)
    lstm_full = {
        'model_name': 'LSTM Autoencoder',
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

    with open(results_dir / 'FD001_lstm_autoencoder_results.json', 'w') as f:
        json.dump(lstm_full, f, indent=2, default=str)

    # Threshold sensitivity table
    thresh_rows = []
    for t_name, t_res in all_threshold_results.items():
        thresh_rows.append({'threshold_strategy': t_name, **t_res})
    pd.DataFrame(thresh_rows).to_csv(
        out_dir / 'FD001_lstm_autoencoder_threshold_sensitivity.csv', index=False
    )

    print("\n" + "=" * 60)
    print("LSTM AUTOENCODER THRESHOLD SENSITIVITY")
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
    print("PHASE 7 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    run_experiment()

