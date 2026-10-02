"""Statistical significance and ablation study experiments.

1. Multi-seed stability: Run all models on FD001 with 5 seeds, report mean +/- std.
2. Ablation: LSTM-AE sequence length sensitivity.
3. Statistical tests: Wilcoxon signed-rank or paired t-test between model pairs.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import DataPipeline
from src.preprocessing.sequences import create_sequences, extract_normal_data
from src.models.baselines import StatisticalThresholdDetector, IsolationForestDetector
from src.models.autoencoder import AutoencoderDetector
from src.models.lstm_autoencoder import LSTMAutoencoderDetector
from src.detection.thresholding import compute_all_thresholds
from src.evaluation.metrics import (
    compute_classification_metrics, create_binary_labels_from_rul,
    compute_early_warning_metrics
)
from src.utils.reproducibility import set_seed, get_device


SEEDS = [42, 123, 456, 789, 1024]
ANOMALY_RUL = 30
PERSISTENCE = 5


def evaluate_single_run(
    detector, test_X, test_ruls, test_eids, test_cycles,
    val_X, val_ruls, threshold_name='percentile_95'
):
    """Score, threshold, and evaluate a fitted detector."""
    val_scores = detector.score(val_X)
    val_labels = create_binary_labels_from_rul(val_ruls, ANOMALY_RUL)
    thresholds = compute_all_thresholds(val_scores, val_labels)
    thresh = thresholds[threshold_name]

    test_scores = detector.score(test_X)
    test_labels = create_binary_labels_from_rul(test_ruls, ANOMALY_RUL)
    preds = (test_scores >= thresh).astype(int)

    cls = compute_classification_metrics(test_labels, preds, test_scores)
    ew = compute_early_warning_metrics(
        test_eids, test_cycles, preds, test_ruls, PERSISTENCE
    )
    
    # Also get best_f1_val threshold results
    best_thresh = thresholds.get('best_f1_val', thresh)
    best_preds = (test_scores >= best_thresh).astype(int)
    best_cls = compute_classification_metrics(test_labels, best_preds, test_scores)
    best_ew = compute_early_warning_metrics(
        test_eids, test_cycles, best_preds, test_ruls, PERSISTENCE
    )

    return {
        'f1': cls['f1'],
        'precision': cls['precision'],
        'recall': cls['recall'],
        'roc_auc': cls.get('roc_auc', float('nan')),
        'pr_auc': cls.get('pr_auc', float('nan')),
        'false_alarm_rate': cls['false_alarm_rate'],
        'detection_rate': ew['detection_rate'],
        'mean_lead_time': ew['mean_lead_time'],
        'f1_best': best_cls['f1'],
        'detection_rate_best': best_ew['detection_rate'],
        'mean_lead_time_best': best_ew['mean_lead_time'],
    }


def run_multi_seed_experiment():
    """Run all models across 5 seeds on FD001."""
    print("\n" + "=" * 70)
    print("EXPERIMENT: MULTI-SEED STATISTICAL STABILITY (5 seeds x 4 models)")
    print("=" * 70)

    device = get_device()
    all_seed_results = {
        'Statistical_mean': [],
        'IsolationForest': [],
        'Autoencoder_FC': [],
        'LSTM_Autoencoder': [],
    }

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        pipeline = DataPipeline({
            'experiment': {'seed': seed},
            'data': {'dataset': 'FD001', 'data_dir': 'data/raw', 'output_dir': 'data/processed'},
            'features': {'selected_sensors': None, 'include_operational_settings': True},
            'training': {'normal_ratio': 0.7},
        })
        data = pipeline.prepare(verbose=False)
        fc = data['feature_cols']

        train_norm_X = data['train_normal_df'][fc].values
        val_X = data['val_df'][fc].values
        val_ruls = data['val_df']['RUL'].values
        test_X = data['test_df'][fc].values
        test_ruls = data['test_df']['RUL'].values
        test_eids = data['test_df']['unit_id'].values
        test_cycles = data['test_df']['cycle'].values

        val_normal_df = extract_normal_data(data['val_df'], normal_ratio=0.7)
        val_norm_X = val_normal_df[fc].values

        # Sequences
        seq_data = pipeline.prepare_sequences(data, sequence_length=30, stride=1)
        train_seqs = seq_data['train_sequences']
        val_seqs = seq_data['val_sequences']
        val_seq_ruls = seq_data['val_ruls']
        test_seqs = seq_data['test_sequences']
        test_seq_ruls = seq_data['test_ruls']
        test_seq_eids = seq_data['test_engine_ids']
        test_seq_cycles = seq_data['test_cycles']

        val_norm_seqs, _, _ = create_sequences(val_normal_df, fc, sequence_length=30, stride=1)

        # 1. Statistical
        print(f"  Statistical...", end=" ", flush=True)
        stat = StatisticalThresholdDetector('mean').fit(train_norm_X)
        r = evaluate_single_run(stat, test_X, test_ruls, test_eids, test_cycles, val_X, val_ruls)
        all_seed_results['Statistical_mean'].append(r)
        print(f"F1={r['f1']:.3f}")

        # 2. Isolation Forest
        print(f"  IsolationForest...", end=" ", flush=True)
        iforest = IsolationForestDetector(n_estimators=200, random_state=seed).fit(train_norm_X)
        r = evaluate_single_run(iforest, test_X, test_ruls, test_eids, test_cycles, val_X, val_ruls)
        all_seed_results['IsolationForest'].append(r)
        print(f"F1={r['f1']:.3f}")

        # 3. FC-Autoencoder
        print(f"  FC-Autoencoder...", end=" ", flush=True)
        ae = AutoencoderDetector(
            input_dim=len(fc), encoder_dims=[64, 32], latent_dim=16,
            max_epochs=60, patience=10, device=device, seed=seed
        )
        ae.fit(train_norm_X, val_norm_X)
        r = evaluate_single_run(ae, test_X, test_ruls, test_eids, test_cycles, val_X, val_ruls)
        all_seed_results['Autoencoder_FC'].append(r)
        print(f"F1={r['f1']:.3f}")

        # 4. LSTM-Autoencoder
        print(f"  LSTM-Autoencoder...", end=" ", flush=True)
        lstm = LSTMAutoencoderDetector(
            input_dim=len(fc), seq_len=30, hidden_dim=64, latent_dim=32,
            num_layers=2, max_epochs=60, patience=10, device=device, seed=seed
        )
        lstm.fit(train_seqs, val_norm_seqs)
        r = evaluate_single_run(
            lstm, test_seqs, test_seq_ruls, test_seq_eids, test_seq_cycles,
            val_seqs, val_seq_ruls
        )
        all_seed_results['LSTM_Autoencoder'].append(r)
        print(f"F1={r['f1']:.3f}")

    return all_seed_results


def run_sequence_length_ablation():
    """Ablation: LSTM-AE performance vs. sequence length."""
    print("\n" + "=" * 70)
    print("ABLATION: LSTM-AE SEQUENCE LENGTH SENSITIVITY")
    print("=" * 70)

    device = get_device()
    seed = 42
    set_seed(seed)
    seq_lengths = [10, 20, 30, 50]
    ablation_results = []

    pipeline = DataPipeline({
        'experiment': {'seed': seed},
        'data': {'dataset': 'FD001', 'data_dir': 'data/raw', 'output_dir': 'data/processed'},
        'features': {'selected_sensors': None, 'include_operational_settings': True},
        'training': {'normal_ratio': 0.7},
    })
    data = pipeline.prepare(verbose=False)
    fc = data['feature_cols']
    val_normal_df = extract_normal_data(data['val_df'], normal_ratio=0.7)

    for sl in seq_lengths:
        print(f"\n  Sequence length = {sl}")
        set_seed(seed)

        seq_data = pipeline.prepare_sequences(data, sequence_length=sl, stride=1)
        train_seqs = seq_data['train_sequences']
        val_seqs = seq_data['val_sequences']
        val_seq_ruls = seq_data['val_ruls']
        test_seqs = seq_data['test_sequences']
        test_seq_ruls = seq_data['test_ruls']
        test_seq_eids = seq_data['test_engine_ids']
        test_seq_cycles = seq_data['test_cycles']

        val_norm_seqs, _, _ = create_sequences(val_normal_df, fc, sequence_length=sl, stride=1)

        lstm = LSTMAutoencoderDetector(
            input_dim=len(fc), seq_len=sl, hidden_dim=64, latent_dim=32,
            num_layers=2, max_epochs=60, patience=10, device=device, seed=seed
        )
        lstm.fit(train_seqs, val_norm_seqs)

        r = evaluate_single_run(
            lstm, test_seqs, test_seq_ruls, test_seq_eids, test_seq_cycles,
            val_seqs, val_seq_ruls
        )
        r['sequence_length'] = sl
        r['n_parameters'] = lstm.n_parameters_
        r['fit_time'] = lstm.fit_time_
        r['best_epoch'] = lstm.best_epoch_
        r['n_train_sequences'] = len(train_seqs)
        ablation_results.append(r)

        print(f"    F1={r['f1']:.3f}, ROC-AUC={r['roc_auc']:.3f}, "
              f"Lead={r['mean_lead_time']:.1f}, Params={r['n_parameters']}")

    return ablation_results


def compute_statistical_tests(all_seed_results):
    """Paired Wilcoxon signed-rank tests between model pairs on F1 scores."""
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTS (Wilcoxon signed-rank on F1)")
    print("=" * 70)

    model_names = list(all_seed_results.keys())
    n_models = len(model_names)

    test_results = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            m1, m2 = model_names[i], model_names[j]
            f1_a = [r['f1'] for r in all_seed_results[m1]]
            f1_b = [r['f1'] for r in all_seed_results[m2]]

            # Paired test
            if len(set(np.array(f1_a) - np.array(f1_b))) > 1:
                stat_w, p_wilcoxon = stats.wilcoxon(f1_a, f1_b)
            else:
                stat_w, p_wilcoxon = float('nan'), 1.0

            stat_t, p_ttest = stats.ttest_rel(f1_a, f1_b)

            diff = np.mean(f1_a) - np.mean(f1_b)
            test_results.append({
                'Model_A': m1,
                'Model_B': m2,
                'Mean_F1_A': np.mean(f1_a),
                'Mean_F1_B': np.mean(f1_b),
                'Diff(A-B)': diff,
                'Wilcoxon_p': p_wilcoxon,
                'Paired_t_p': p_ttest,
                'Significant_0.05': 'Yes' if p_ttest < 0.05 else 'No',
            })

            sig = "*" if p_ttest < 0.05 else "ns"
            print(f"  {m1} vs {m2}: diff={diff:+.4f}, p={p_ttest:.4f} [{sig}]")

    return test_results


def main():
    start = time.time()

    # 1. Multi-seed stability
    all_seed_results = run_multi_seed_experiment()

    # Summary table with mean +/- std
    print("\n" + "=" * 70)
    print("MULTI-SEED SUMMARY TABLE (mean +/- std across 5 seeds)")
    print("=" * 70)

    summary_rows = []
    metrics_to_report = ['f1', 'roc_auc', 'pr_auc', 'false_alarm_rate', 'detection_rate', 'mean_lead_time']
    for model, results in all_seed_results.items():
        row = {'Model': model}
        for m in metrics_to_report:
            vals = [r[m] for r in results]
            row[f'{m}_mean'] = np.mean(vals)
            row[f'{m}_std'] = np.std(vals)
            row[f'{m}'] = f"{np.mean(vals):.3f} +/- {np.std(vals):.3f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    display_cols = ['Model'] + metrics_to_report
    print(summary_df[display_cols].to_string(index=False))

    # 2. Statistical significance tests
    sig_results = compute_statistical_tests(all_seed_results)

    # 3. Sequence length ablation
    ablation_results = run_sequence_length_ablation()

    print("\n" + "=" * 70)
    print("SEQUENCE LENGTH ABLATION SUMMARY")
    print("=" * 70)
    abl_rows = []
    for r in ablation_results:
        abl_rows.append({
            'SeqLen': r['sequence_length'],
            'Params': r['n_parameters'],
            'F1(p95)': f"{r['f1']:.3f}",
            'ROC-AUC': f"{r['roc_auc']:.3f}",
            'PR-AUC': f"{r['pr_auc']:.3f}",
            'FAR': f"{r['false_alarm_rate']:.3f}",
            'DetRate': f"{r['detection_rate']:.2%}",
            'LeadTime': f"{r['mean_lead_time']:.1f}",
            'F1(best)': f"{r['f1_best']:.3f}",
            'FitTime': f"{r['fit_time']:.1f}s",
            'BestEpoch': r['best_epoch'],
        })
    abl_df = pd.DataFrame(abl_rows)
    print(abl_df.to_string(index=False))

    # Save everything
    out_dir = Path('results/tables/ablation')
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(out_dir / 'multi_seed_summary.csv', index=False)
    pd.DataFrame(sig_results).to_csv(out_dir / 'statistical_tests.csv', index=False)
    abl_df.to_csv(out_dir / 'sequence_length_ablation.csv', index=False)

    # Save raw seed results
    logs_dir = Path('results/logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / 'multi_seed_raw_results.json', 'w') as f:
        json.dump(all_seed_results, f, indent=2, default=str)

    total = time.time() - start
    print(f"\nAll ablation and significance experiments completed in {total/60:.1f} minutes.")
    print(f"Results saved to {out_dir}")

    print("\n" + "=" * 70)
    print("PHASE 10 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
