"""Publication-quality visualization figures for the research paper.

Generates:
1. Model comparison bar chart (F1, ROC-AUC)
2. Threshold sensitivity curves
3. Early warning lead time comparison
4. Cross-condition heatmap
5. Training loss curves
6. Ablation study plots
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

# Publication style
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.figsize': (8, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
})

OUT_DIR = Path('results/figures/publication')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fig1_model_comparison_bar():
    """Bar chart comparing all models on FD001 (percentile_95 threshold)."""
    # FD001 results from our experiments
    models = ['Statistical\n(mean z)', 'Isolation\nForest', 'One-Class\nSVM', 'Autoencoder\n(FC)', 'LSTM\nAutoencoder']
    f1_scores = [0.602, 0.549, 0.588, 0.564, 0.429]
    roc_aucs = [0.984, 0.976, 0.972, 0.912, 0.901]
    pr_aucs = [0.932, 0.905, 0.913, 0.769, 0.689]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, f1_scores, width, label='F1 Score', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x, roc_aucs, width, label='ROC-AUC', color='#4CAF50', alpha=0.85)
    bars3 = ax.bar(x + width, pr_aucs, width, label='PR-AUC', color='#FF9800', alpha=0.85)

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison on C-MAPSS FD001 (percentile_95 threshold)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='lower left')
    ax.set_ylim(0.3, 1.05)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig1_model_comparison_bar.png', bbox_inches='tight')
    plt.close()
    print("  Saved fig1_model_comparison_bar.png")


def fig2_lead_time_comparison():
    """Lead time comparison across models."""
    models = ['Statistical\n(mean z)', 'Isolation\nForest', 'One-Class\nSVM', 'FC-AE', 'LSTM-AE']
    lead_times_p95 = [10.5, 11.1, 11.5, 14.2, 50.0]
    detection_rates = [93.3, 73.3, 86.7, 60.0, 73.3]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    # Lead time
    bars = ax1.bar(models, lead_times_p95, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Mean Lead Time (cycles before failure)')
    ax1.set_title('Early Warning Lead Time')
    for bar, val in zip(bars, lead_times_p95):
        ax1.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
    ax1.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='RUL=30 threshold')
    ax1.legend()

    # Detection rate
    bars = ax2.bar(models, detection_rates, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Detection Rate (%)')
    ax2.set_title('Engine Detection Rate')
    ax2.set_ylim(0, 105)
    for bar, val in zip(bars, detection_rates):
        ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig2_lead_time_detection.png', bbox_inches='tight')
    plt.close()
    print("  Saved fig2_lead_time_detection.png")


def fig3_cross_condition_heatmap():
    """Heatmap of cross-condition transfer performance (ROC-AUC)."""
    csv_path = Path('results/tables/cross_condition/cross_condition_transfer_summary.csv')
    if not csv_path.exists():
        print("  Skipping fig3: cross_condition_transfer_summary.csv not found")
        return

    df = pd.read_csv(csv_path)
    models = df['model'].unique()
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']

    # Build ROC-AUC matrix
    roc_matrix = np.zeros((len(models), len(datasets)))
    for i, m in enumerate(models):
        for j, d in enumerate(datasets):
            row = df[(df['model'] == m) & (df['target_dataset'] == d)]
            if len(row) > 0:
                roc_matrix[i, j] = row['roc_auc'].values[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(roc_matrix, cmap='RdYlGn', vmin=0.4, vmax=1.0, aspect='auto')

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_yticks(range(len(models)))
    model_labels = ['Statistical (mean z)', 'Isolation Forest', 'FC-Autoencoder', 'LSTM-Autoencoder']
    ax.set_yticklabels(model_labels)

    ax.set_xlabel('Target Dataset (Evaluated On)')
    ax.set_ylabel('Model (Trained on FD001)')
    ax.set_title('Zero-Shot Cross-Condition Transfer: ROC-AUC')

    for i in range(len(models)):
        for j in range(len(datasets)):
            val = roc_matrix[i, j]
            color = 'white' if val < 0.65 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=11, fontweight='bold')

    plt.colorbar(im, ax=ax, label='ROC-AUC')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig3_cross_condition_heatmap.png', bbox_inches='tight')
    plt.close()
    print("  Saved fig3_cross_condition_heatmap.png")


def fig4_within_dataset_grouped():
    """Grouped bar chart for within-dataset complexity benchmark."""
    csv_path = Path('results/tables/cross_condition/within_dataset_complexity_summary.csv')
    if not csv_path.exists():
        print("  Skipping fig4: within_dataset_complexity_summary.csv not found")
        return

    df = pd.read_csv(csv_path)
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    models = ['Statistical_mean', 'IsolationForest', 'Autoencoder_FC', 'LSTM_Autoencoder']
    model_labels = ['Statistical', 'Isol. Forest', 'FC-AE', 'LSTM-AE']
    colors = ['#2196F3', '#4CAF50', '#9C27B0', '#F44336']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC-AUC
    x = np.arange(len(datasets))
    width = 0.2
    for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        vals = []
        for d in datasets:
            row = df[(df['model'] == model) & (df['dataset'] == d)]
            vals.append(row['roc_auc'].values[0] if len(row) > 0 else 0)
        axes[0].bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    axes[0].set_ylabel('ROC-AUC')
    axes[0].set_title('ROC-AUC Across Dataset Complexity')
    axes[0].set_xticks(x + 1.5 * width)
    axes[0].set_xticklabels(['FD001\n1 cond, 1 fault', 'FD002\n6 cond, 1 fault',
                              'FD003\n1 cond, 2 faults', 'FD004\n6 cond, 2 faults'])
    axes[0].legend()
    axes[0].set_ylim(0.3, 1.05)
    axes[0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    # F1
    for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        vals = []
        for d in datasets:
            row = df[(df['model'] == model) & (df['dataset'] == d)]
            vals.append(row['f1'].values[0] if len(row) > 0 else 0)
        axes[1].bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('F1 Across Dataset Complexity')
    axes[1].set_xticks(x + 1.5 * width)
    axes[1].set_xticklabels(['FD001\n1 cond, 1 fault', 'FD002\n6 cond, 1 fault',
                              'FD003\n1 cond, 2 faults', 'FD004\n6 cond, 2 faults'])
    axes[1].legend()
    axes[1].set_ylim(0, 0.7)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig4_within_dataset_complexity.png', bbox_inches='tight')
    plt.close()
    print("  Saved fig4_within_dataset_complexity.png")


def fig5_complexity_vs_method_radar():
    """Summary table figure showing method recommendation by scenario."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    scenarios = [
        ['Scenario', 'Best Model', 'Why', 'Key Metric'],
        ['Single condition, simple', 'Statistical (mean z)', 'Zero cost, highest AUC', 'ROC-AUC: 0.984'],
        ['Novel fault modes', 'Isolation Forest', 'Robust to manifold shift', 'F1: 0.596 (transfer)'],
        ['Multi-condition fleet', 'FC-Autoencoder', 'Learns non-linear regimes', 'ROC-AUC: 0.964 (FD004)'],
        ['Maximum lead time', 'LSTM Autoencoder', 'Temporal trajectory drift', 'Lead: 50-111 cycles'],
        ['Edge / real-time', 'Isolation Forest', 'Fast inference, no GPU', 'Score: <0.01ms/sample'],
    ]

    table = ax.table(cellText=scenarios[1:], colLabels=scenarios[0],
                     cellLoc='center', loc='center',
                     colColours=['#E3F2FD'] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    ax.set_title('Method Selection Guide: Which Model to Deploy?', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig5_method_selection_guide.png', bbox_inches='tight')
    plt.close()
    print("  Saved fig5_method_selection_guide.png")


def fig6_training_curves():
    """Training loss curves for AE and LSTM-AE."""
    ae_path = Path('results/logs/FD001_autoencoder_results.json')
    lstm_path = Path('results/logs/FD001_lstm_autoencoder_results.json')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if ae_path.exists():
        with open(ae_path) as f:
            ae = json.load(f)
        ax1.plot(ae['train_losses'], label='Train', color='#2196F3', linewidth=1.5)
        ax1.plot(ae['val_losses'], label='Validation', color='#F44336', linewidth=1.5)
        best = ae['model_params']['best_epoch']
        ax1.axvline(x=best - 1, color='green', linestyle='--', alpha=0.7, label=f'Best epoch ({best})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('FC-Autoencoder Training')
        ax1.legend()

    if lstm_path.exists():
        with open(lstm_path) as f:
            lstm = json.load(f)
        ax2.plot(lstm['train_losses'], label='Train', color='#2196F3', linewidth=1.5)
        ax2.plot(lstm['val_losses'], label='Validation', color='#F44336', linewidth=1.5)
        best = lstm['model_params']['best_epoch']
        ax2.axvline(x=best - 1, color='green', linestyle='--', alpha=0.7, label=f'Best epoch ({best})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE Loss')
        ax2.set_title('LSTM-Autoencoder Training')
        ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig6_training_curves.png', bbox_inches='tight')
    plt.close()
    print("  Saved fig6_training_curves.png")


def main():
    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    fig1_model_comparison_bar()
    fig2_lead_time_comparison()
    fig3_cross_condition_heatmap()
    fig4_within_dataset_grouped()
    fig5_complexity_vs_method_radar()
    fig6_training_curves()

    print(f"\nAll figures saved to {OUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
