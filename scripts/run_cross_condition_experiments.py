"""Cross-condition and cross-subset generalization experiments for C-MAPSS.

Investigates:
- RQ3: How model performance changes across operating conditions and fault modes (FD001 -> FD004).
- RQ6: Cross-condition transfer robustness (models trained on FD001 evaluated directly on FD002, FD003, FD004).
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cmapss_loader import load_train_data
from src.preprocessing.pipeline import DataPipeline
from src.preprocessing.sequences import create_sequences, create_sequences_with_rul, extract_normal_data
from src.preprocessing.normalization import SafeScaler
from src.models.baselines import StatisticalThresholdDetector, IsolationForestDetector, OneClassSVMDetector
from src.models.autoencoder import AutoencoderDetector
from src.models.lstm_autoencoder import LSTMAutoencoderDetector
from src.detection.thresholding import compute_all_thresholds
from src.evaluation.metrics import (
    compute_classification_metrics, create_binary_labels_from_rul,
    compute_early_warning_metrics
)
from src.utils.reproducibility import set_seed, get_device


def evaluate_model_on_data(
    model_name: str,
    detector: Any,
    test_X: np.ndarray,
    test_ruls: np.ndarray,
    test_eids: np.ndarray,
    test_cycles: np.ndarray,
    threshold_value: float,
    threshold_name: str = 'percentile_95',
    is_sequence: bool = False,
    anomaly_rul: int = 30,
    persistence: int = 5
) -> Dict[str, Any]:
    """Score and evaluate a fitted model on target test data."""
    start = time.time()
    scores = detector.score(test_X)
    score_time = time.time() - start

    test_labels = create_binary_labels_from_rul(test_ruls, anomaly_rul)
    preds = (scores >= threshold_value).astype(int)

    cls_metrics = compute_classification_metrics(test_labels, preds, scores)
    ew_metrics = compute_early_warning_metrics(
        test_eids, test_cycles, preds, test_ruls, persistence
    )

    return {
        'model': model_name,
        'threshold_name': threshold_name,
        'threshold_value': threshold_value,
        'precision': cls_metrics['precision'],
        'recall': cls_metrics['recall'],
        'f1': cls_metrics['f1'],
        'false_alarm_rate': cls_metrics['false_alarm_rate'],
        'roc_auc': cls_metrics.get('roc_auc', float('nan')),
        'pr_auc': cls_metrics.get('pr_auc', float('nan')),
        'detection_rate': ew_metrics['detection_rate'],
        'mean_lead_time': ew_metrics['mean_lead_time'],
        'score_time_seconds': score_time,
    }


def run_cross_condition_transfer():
    """Experiment 1: Train on FD001 -> Test on FD002, FD003, FD004 (RQ6)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: ZERO-SHOT CROSS-CONDITION TRANSFER (RQ6)")
    print("Training on FD001 (1 condition, 1 fault) -> Testing on FD002, FD003, FD004")
    print("=" * 70)

    seed = 42
    set_seed(seed)
    device = get_device()

    # 1. Prepare FD001 source pipeline
    pipeline_fd001 = DataPipeline({
        'experiment': {'seed': seed},
        'data': {'dataset': 'FD001', 'data_dir': 'data/raw', 'output_dir': 'data/processed'},
        'features': {'selected_sensors': None, 'include_operational_settings': True},
        'training': {'normal_ratio': 0.7},
    })
    data_fd001 = pipeline_fd001.prepare(verbose=False)
    feature_cols = data_fd001['feature_cols']
    scaler = data_fd001['scaler']

    train_normal_X = data_fd001['train_normal_df'][feature_cols].values
    val_X = data_fd001['val_df'][feature_cols].values
    val_ruls = data_fd001['val_df']['RUL'].values

    # Sequences for LSTM-AE
    seq_data_fd001 = pipeline_fd001.prepare_sequences(data_fd001, sequence_length=30, stride=1)
    train_seqs = seq_data_fd001['train_sequences']
    val_seqs = seq_data_fd001['val_sequences']
    val_ruls_seq = seq_data_fd001['val_ruls']

    val_normal_df = extract_normal_data(data_fd001['val_df'], normal_ratio=0.7)
    val_normal_seqs, _, _ = create_sequences(val_normal_df, feature_cols, sequence_length=30, stride=1)

    # 2. Train all 4 models on FD001
    print("\nTraining models on FD001...")
    models = {}
    thresholds = {}

    # Statistical (mean z)
    print("  Fitting Statistical (mean z)...")
    stat_detector = StatisticalThresholdDetector('mean')
    stat_detector.fit(train_normal_X)
    stat_val_scores = stat_detector.score(val_X)
    stat_thresh = compute_all_thresholds(stat_val_scores)['percentile_95']
    models['Statistical_mean'] = (stat_detector, stat_thresh, False)

    # Isolation Forest
    print("  Fitting Isolation Forest...")
    if_detector = IsolationForestDetector(n_estimators=200, random_state=seed)
    if_detector.fit(train_normal_X)
    if_val_scores = if_detector.score(val_X)
    if_thresh = compute_all_thresholds(if_val_scores)['percentile_95']
    models['IsolationForest'] = (if_detector, if_thresh, False)

    # FC-Autoencoder
    print("  Fitting FC-Autoencoder...")
    ae_detector = AutoencoderDetector(
        input_dim=len(feature_cols), encoder_dims=[64, 32], latent_dim=16,
        max_epochs=60, patience=10, device=device, seed=seed
    )
    val_normal_X = val_normal_df[feature_cols].values
    ae_detector.fit(train_normal_X, val_normal_X)
    ae_val_scores = ae_detector.score(val_X)
    ae_thresh = compute_all_thresholds(ae_val_scores)['percentile_95']
    models['Autoencoder_FC'] = (ae_detector, ae_thresh, False)

    # LSTM-Autoencoder
    print("  Fitting LSTM-Autoencoder...")
    lstm_detector = LSTMAutoencoderDetector(
        input_dim=len(feature_cols), seq_len=30, hidden_dim=64, latent_dim=32,
        num_layers=2, max_epochs=60, patience=10, device=device, seed=seed
    )
    lstm_detector.fit(train_seqs, val_normal_seqs)
    lstm_val_scores = lstm_detector.score(val_seqs)
    lstm_thresh = compute_all_thresholds(lstm_val_scores)['percentile_95']
    models['LSTM_Autoencoder'] = (lstm_detector, lstm_thresh, True)

    # 3. Evaluate each model on FD001 (baseline), FD002, FD003, FD004 test splits
    target_subsets = ['FD001', 'FD002', 'FD003', 'FD004']
    transfer_results = []

    for subset in target_subsets:
        print(f"\nEvaluating models on {subset}...")
        # Load raw train data to get complete engine trajectories for evaluation
        target_raw = load_train_data('data/raw', subset)
        # Use target's test engines (e.g. engines outside training ratio) or target test set
        # To maintain strict evaluation, split target into 70/15/15 and use the 15% test engines
        _, _, target_test, _ = pipeline_fd001.config['data'].copy(), None, None, None
        
        # Split target data by engine ID
        from src.preprocessing.splitting import engine_level_split
        _, _, target_test_df, _ = engine_level_split(
            target_raw, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed
        )

        # Normalize target data using FD001's scaler (crucial for cross-condition test)
        # Check if all required features are present
        target_norm_df = scaler.transform(target_test_df)

        target_X = target_norm_df[feature_cols].values
        target_ruls = target_norm_df['RUL'].values
        target_eids = target_norm_df['unit_id'].values
        target_cycles = target_norm_df['cycle'].values

        # Sequence data for LSTM-AE
        target_seqs, seq_eids, seq_cycles, seq_ruls = create_sequences_with_rul(
            target_norm_df, feature_cols, sequence_length=30, stride=1
        )

        for model_name, (detector, threshold_val, is_seq) in models.items():
            if is_seq:
                res = evaluate_model_on_data(
                    model_name=model_name,
                    detector=detector,
                    test_X=target_seqs,
                    test_ruls=seq_ruls,
                    test_eids=seq_eids,
                    test_cycles=seq_cycles,
                    threshold_value=threshold_val,
                    is_sequence=True
                )
            else:
                res = evaluate_model_on_data(
                    model_name=model_name,
                    detector=detector,
                    test_X=target_X,
                    test_ruls=target_ruls,
                    test_eids=target_eids,
                    test_cycles=target_cycles,
                    threshold_value=threshold_val,
                    is_sequence=False
                )
            res['target_dataset'] = subset
            transfer_results.append(res)

    transfer_df = pd.DataFrame(transfer_results)
    
    # Reorder columns
    cols = ['model', 'target_dataset', 'f1', 'precision', 'recall', 'false_alarm_rate', 'roc_auc', 'detection_rate', 'mean_lead_time']
    transfer_summary = transfer_df[cols]

    print("\n" + "=" * 70)
    print("CROSS-DATASET GENERALIZATION RESULTS (FD001 Model -> Target Datasets)")
    print("=" * 70)
    print(transfer_summary.to_string(index=False))

    out_dir = Path('results/tables/cross_condition')
    out_dir.mkdir(parents=True, exist_ok=True)
    transfer_summary.to_csv(out_dir / 'cross_condition_transfer_summary.csv', index=False)

    return transfer_df


def run_within_subset_comparison():
    """Experiment 2: Within-dataset performance across subsets FD001, FD002, FD003, FD004 (RQ3)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: WITHIN-DATASET OPERATING COMPLEXITY BENCHMARK (RQ3)")
    print("Evaluating how each dataset's intrinsic complexity impacts anomaly detection")
    print("=" * 70)

    seed = 42
    set_seed(seed)
    device = get_device()

    subsets = ['FD001', 'FD002', 'FD003', 'FD004']
    within_results = []

    for subset in subsets:
        print(f"\n{'='*40}\nProcessing dataset: {subset}\n{'='*40}")
        pipeline = DataPipeline({
            'experiment': {'seed': seed},
            'data': {'dataset': subset, 'data_dir': 'data/raw', 'output_dir': 'data/processed'},
            'features': {'selected_sensors': None, 'include_operational_settings': True},
            'training': {'normal_ratio': 0.7},
        })
        data = pipeline.prepare(verbose=False)
        feature_cols = data['feature_cols']

        train_norm_X = data['train_normal_df'][feature_cols].values
        val_X = data['val_df'][feature_cols].values
        val_ruls = data['val_df']['RUL'].values

        test_X = data['test_df'][feature_cols].values
        test_ruls = data['test_df']['RUL'].values
        test_eids = data['test_df']['unit_id'].values
        test_cycles = data['test_df']['cycle'].values

        # 1. Statistical (mean z)
        stat = StatisticalThresholdDetector('mean').fit(train_norm_X)
        val_scores = stat.score(val_X)
        thresh = compute_all_thresholds(val_scores)['percentile_95']
        res = evaluate_model_on_data('Statistical_mean', stat, test_X, test_ruls, test_eids, test_cycles, thresh)
        res['dataset'] = subset
        within_results.append(res)

        # 2. Isolation Forest
        iforest = IsolationForestDetector(n_estimators=200, random_state=seed).fit(train_norm_X)
        val_scores = iforest.score(val_X)
        thresh = compute_all_thresholds(val_scores)['percentile_95']
        res = evaluate_model_on_data('IsolationForest', iforest, test_X, test_ruls, test_eids, test_cycles, thresh)
        res['dataset'] = subset
        within_results.append(res)

        # 3. FC-Autoencoder
        ae = AutoencoderDetector(
            input_dim=len(feature_cols), encoder_dims=[64, 32], latent_dim=16,
            max_epochs=40, patience=8, device=device, seed=seed
        )
        val_normal_df = extract_normal_data(data['val_df'], normal_ratio=0.7)
        val_norm_X = val_normal_df[feature_cols].values
        ae.fit(train_norm_X, val_norm_X)
        val_scores = ae.score(val_X)
        thresh = compute_all_thresholds(val_scores)['percentile_95']
        res = evaluate_model_on_data('Autoencoder_FC', ae, test_X, test_ruls, test_eids, test_cycles, thresh)
        res['dataset'] = subset
        within_results.append(res)

        # 4. LSTM Autoencoder
        seq_data = pipeline.prepare_sequences(data, sequence_length=30, stride=1)
        train_seqs = seq_data['train_sequences']
        val_seqs = seq_data['val_sequences']
        test_seqs = seq_data['test_sequences']
        test_s_ruls = seq_data['test_ruls']
        test_s_eids = seq_data['test_engine_ids']
        test_s_cycles = seq_data['test_cycles']

        val_norm_seqs, _, _ = create_sequences(val_normal_df, feature_cols, sequence_length=30, stride=1)

        lstm = LSTMAutoencoderDetector(
            input_dim=len(feature_cols), seq_len=30, hidden_dim=64, latent_dim=32,
            num_layers=2, max_epochs=40, patience=8, device=device, seed=seed
        )
        lstm.fit(train_seqs, val_norm_seqs)
        val_scores = lstm.score(val_seqs)
        thresh = compute_all_thresholds(val_scores)['percentile_95']
        res = evaluate_model_on_data(
            'LSTM_Autoencoder', lstm, test_seqs, test_s_ruls, test_s_eids, test_s_cycles, thresh, is_sequence=True
        )
        res['dataset'] = subset
        within_results.append(res)

    within_df = pd.DataFrame(within_results)
    cols = ['dataset', 'model', 'f1', 'precision', 'recall', 'false_alarm_rate', 'roc_auc', 'detection_rate', 'mean_lead_time']
    within_summary = within_df[cols]

    print("\n" + "=" * 70)
    print("WITHIN-DATASET OPERATING COMPLEXITY BENCHMARK (FD001 - FD004)")
    print("=" * 70)
    print(within_summary.to_string(index=False))

    out_dir = Path('results/tables/cross_condition')
    out_dir.mkdir(parents=True, exist_ok=True)
    within_summary.to_csv(out_dir / 'within_dataset_complexity_summary.csv', index=False)

    return within_df


def main():
    start_total = time.time()
    
    # 1. Cross-condition transfer (RQ6)
    transfer_df = run_cross_condition_transfer()

    # 2. Within-dataset complexity (RQ3)
    within_df = run_within_subset_comparison()

    total_time = time.time() - start_total
    print(f"\nAll cross-condition and complexity experiments completed in {total_time/60:.1f} minutes.")


if __name__ == '__main__':
    main()

