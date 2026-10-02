"""Comprehensive Exploratory Data Analysis for C-MAPSS dataset.

Generates all required analyses:
1. Dataset description & feature dictionary
2. Missing-value analysis
3. Constant-feature analysis
4. Correlation analysis
5. Sensor-distribution analysis
6. Operating-condition analysis
7. Engine lifetime analysis
8. Degradation trend analysis
9. Representative engine trajectory visualization

All figures saved to results/figures/eda/
All tables saved to results/tables/eda/
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.cmapss_loader import (
    load_train_data, load_test_data, get_dataset_summary,
    COLUMN_NAMES, SENSOR_DESCRIPTIONS, OPERATIONAL_SETTING_DESCRIPTIONS,
    DATASET_INFO
)

# Visualization settings
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (12, 8),
    'savefig.bbox': 'tight',
})

warnings.filterwarnings('ignore', category=FutureWarning)


def ensure_dirs(fig_dir: Path, table_dir: Path):
    """Create output directories."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)


def analyze_feature_dictionary(train_df: pd.DataFrame, table_dir: Path, subset: str):
    """Create and save feature dictionary."""
    sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]
    op_cols = [c for c in train_df.columns if c.startswith('op_setting_')]
    
    rows = []
    # ID columns
    rows.append({'Feature': 'unit_id', 'Type': 'ID', 'Description': 'Engine unit identifier',
                 'Min': train_df['unit_id'].min(), 'Max': train_df['unit_id'].max(),
                 'Mean': '-', 'Std': '-', 'Unique': train_df['unit_id'].nunique()})
    rows.append({'Feature': 'cycle', 'Type': 'Time', 'Description': 'Engine operating cycle',
                 'Min': train_df['cycle'].min(), 'Max': train_df['cycle'].max(),
                 'Mean': f"{train_df['cycle'].mean():.1f}", 'Std': f"{train_df['cycle'].std():.1f}",
                 'Unique': train_df['cycle'].nunique()})
    
    # Operational settings
    for col in op_cols:
        desc = OPERATIONAL_SETTING_DESCRIPTIONS.get(col, 'Unknown')
        rows.append({
            'Feature': col, 'Type': 'Operational',
            'Description': desc,
            'Min': f"{train_df[col].min():.4f}",
            'Max': f"{train_df[col].max():.4f}",
            'Mean': f"{train_df[col].mean():.4f}",
            'Std': f"{train_df[col].std():.4f}",
            'Unique': train_df[col].nunique()
        })
    
    # Sensors
    for col in sensor_cols:
        desc = SENSOR_DESCRIPTIONS.get(col, 'Unknown')
        std_val = train_df[col].std()
        rows.append({
            'Feature': col, 'Type': 'Sensor',
            'Description': desc,
            'Min': f"{train_df[col].min():.4f}",
            'Max': f"{train_df[col].max():.4f}",
            'Mean': f"{train_df[col].mean():.4f}",
            'Std': f"{std_val:.4f}",
            'Unique': train_df[col].nunique(),
            'Constant': 'Yes' if std_val == 0 else 'No'
        })
    
    # RUL
    rows.append({
        'Feature': 'RUL', 'Type': 'Target',
        'Description': 'Remaining Useful Life (computed)',
        'Min': train_df['RUL'].min(), 'Max': train_df['RUL'].max(),
        'Mean': f"{train_df['RUL'].mean():.1f}", 'Std': f"{train_df['RUL'].std():.1f}",
        'Unique': train_df['RUL'].nunique()
    })
    
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(table_dir / f'{subset}_feature_dictionary.csv', index=False)
    print(f"  Feature dictionary saved ({len(rows)} features)")
    return feat_df


def analyze_constant_features(train_df: pd.DataFrame, table_dir: Path, subset: str):
    """Identify and document constant/near-constant features."""
    sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]
    op_cols = [c for c in train_df.columns if c.startswith('op_setting_')]
    
    results = []
    for col in op_cols + sensor_cols:
        std = train_df[col].std()
        n_unique = train_df[col].nunique()
        value_range = train_df[col].max() - train_df[col].min()
        cv = std / abs(train_df[col].mean()) if train_df[col].mean() != 0 else 0
        
        status = 'Constant' if std == 0 else ('Near-constant' if n_unique <= 5 else 'Variable')
        
        results.append({
            'Feature': col,
            'Std': f"{std:.6f}",
            'N_Unique': n_unique,
            'Range': f"{value_range:.6f}",
            'CV': f"{cv:.6f}",
            'Status': status,
            'Description': SENSOR_DESCRIPTIONS.get(col, OPERATIONAL_SETTING_DESCRIPTIONS.get(col, ''))
        })
    
    const_df = pd.DataFrame(results)
    const_df.to_csv(table_dir / f'{subset}_constant_analysis.csv', index=False)
    
    constant = const_df[const_df['Status'] == 'Constant']['Feature'].tolist()
    near_const = const_df[const_df['Status'] == 'Near-constant']['Feature'].tolist()
    variable = const_df[const_df['Status'] == 'Variable']['Feature'].tolist()
    
    print(f"  Constant features: {constant}")
    print(f"  Near-constant features: {near_const}")
    print(f"  Variable features: {len(variable)}")
    
    return constant, near_const, variable


def plot_correlation_matrix(train_df: pd.DataFrame, variable_features: list,
                            fig_dir: Path, subset: str):
    """Generate sensor correlation matrix."""
    corr = train_df[variable_features].corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                annot_kws={'size': 7})
    ax.set_title(f'C-MAPSS {subset} - Sensor Correlation Matrix\n(Variable features only)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / f'{subset}_correlation_matrix.png')
    plt.close()
    
    # Save correlation with RUL
    if 'RUL' in train_df.columns:
        rul_corr = train_df[variable_features + ['RUL']].corr()['RUL'].drop('RUL').sort_values()
        rul_corr_df = pd.DataFrame({
            'Feature': rul_corr.index,
            'Correlation_with_RUL': rul_corr.values
        })
        rul_corr_df.to_csv(fig_dir.parent.parent / 'tables' / 'eda' / f'{subset}_rul_correlation.csv', 
                           index=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#d9534f' if v < 0 else '#5cb85c' for v in rul_corr.values]
        ax.barh(rul_corr.index, rul_corr.values, color=colors)
        ax.set_xlabel('Pearson Correlation with RUL')
        ax.set_title(f'C-MAPSS {subset} - Sensor Correlation with RUL', 
                      fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(fig_dir / f'{subset}_rul_correlation.png')
        plt.close()
    
    print(f"  Correlation matrix saved")


def plot_sensor_distributions(train_df: pd.DataFrame, variable_features: list,
                               fig_dir: Path, subset: str):
    """Plot sensor value distributions."""
    sensor_features = [f for f in variable_features if f.startswith('sensor_')]
    
    n_sensors = len(sensor_features)
    n_cols = 4
    n_rows = (n_sensors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(sensor_features):
        ax = axes[i]
        ax.hist(train_df[col], bins=50, alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        
        # Add mean/std annotation
        mean_val = train_df[col].mean()
        std_val = train_df[col].std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, alpha=0.8)
        ax.text(0.95, 0.95, f'u={mean_val:.1f}\ns={std_val:.2f}', 
                transform=ax.transAxes, fontsize=7, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide empty subplots
    for j in range(n_sensors, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(f'C-MAPSS {subset} - Sensor Value Distributions (Training Data)', 
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(fig_dir / f'{subset}_sensor_distributions.png')
    plt.close()
    print(f"  Sensor distributions saved")


def plot_engine_lifetimes(train_df: pd.DataFrame, fig_dir: Path, subset: str):
    """Plot engine lifetime analysis."""
    lifetimes = train_df.groupby('unit_id')['cycle'].max().sort_values()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(lifetimes, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0].axvline(lifetimes.mean(), color='red', linestyle='--', label=f'Mean: {lifetimes.mean():.0f}')
    axes[0].axvline(lifetimes.median(), color='orange', linestyle='--', label=f'Median: {lifetimes.median():.0f}')
    axes[0].set_xlabel('Engine Lifetime (cycles)')
    axes[0].set_ylabel('Number of Engines')
    axes[0].set_title(f'{subset} - Engine Lifetime Distribution')
    axes[0].legend()
    
    # Sorted bar
    axes[1].bar(range(len(lifetimes)), lifetimes.values, color='steelblue', alpha=0.7)
    axes[1].set_xlabel('Engine (sorted by lifetime)')
    axes[1].set_ylabel('Lifetime (cycles)')
    axes[1].set_title(f'{subset} - Sorted Engine Lifetimes')
    axes[1].axhline(lifetimes.mean(), color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(fig_dir / f'{subset}_engine_lifetimes.png')
    plt.close()
    print(f"  Engine lifetime analysis saved")


def plot_degradation_trajectories(train_df: pd.DataFrame, variable_features: list,
                                    fig_dir: Path, subset: str, n_engines: int = 5):
    """Plot degradation trajectories for representative engines."""
    sensor_features = [f for f in variable_features if f.startswith('sensor_')]
    
    # Select representative engines (short, medium, long lifetimes)
    lifetimes = train_df.groupby('unit_id')['cycle'].max().sort_values()
    n_total = len(lifetimes)
    
    # Pick engines at different percentiles
    percentile_indices = [0, n_total // 4, n_total // 2, 3 * n_total // 4, n_total - 1]
    selected_engines = [lifetimes.index[i] for i in percentile_indices]
    
    # Plot 6 key sensors (most correlated with RUL)
    if 'RUL' in train_df.columns:
        rul_corr = train_df[sensor_features + ['RUL']].corr()['RUL'].drop('RUL').abs().sort_values(ascending=False)
        key_sensors = rul_corr.head(6).index.tolist()
    else:
        key_sensors = sensor_features[:6]
    
    fig, axes = plt.subplots(len(key_sensors), 1, figsize=(14, 3 * len(key_sensors)), sharex=False)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_engines)))
    
    for idx, sensor in enumerate(key_sensors):
        ax = axes[idx]
        for eng_idx, engine_id in enumerate(selected_engines):
            eng_data = train_df[train_df['unit_id'] == engine_id]
            ax.plot(eng_data['cycle'], eng_data[sensor], 
                    alpha=0.7, linewidth=0.8, color=colors[eng_idx],
                    label=f'Engine {engine_id} (life={lifetimes[engine_id]})')
        
        desc = SENSOR_DESCRIPTIONS.get(sensor, sensor)
        ax.set_ylabel(sensor)
        ax.set_title(f'{sensor}: {desc}', fontsize=10)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Cycle')
    fig.suptitle(f'C-MAPSS {subset} - Degradation Trajectories\n(Top {len(key_sensors)} sensors by RUL correlation)', 
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(fig_dir / f'{subset}_degradation_trajectories.png')
    plt.close()
    print(f"  Degradation trajectories saved")


def plot_degradation_vs_rul(train_df: pd.DataFrame, variable_features: list,
                              fig_dir: Path, subset: str):
    """Plot sensor values vs RUL to visualize degradation patterns."""
    sensor_features = [f for f in variable_features if f.startswith('sensor_')]
    
    if 'RUL' not in train_df.columns:
        return
    
    # Top 6 sensors by RUL correlation
    rul_corr = train_df[sensor_features + ['RUL']].corr()['RUL'].drop('RUL').abs().sort_values(ascending=False)
    key_sensors = rul_corr.head(6).index.tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, sensor in enumerate(key_sensors):
        ax = axes[idx]
        # Sample for speed
        sample = train_df.sample(min(5000, len(train_df)), random_state=42)
        ax.scatter(sample['RUL'], sample[sensor], alpha=0.1, s=3, c='steelblue')
        
        # Add trend line (binned mean)
        bins = pd.cut(train_df['RUL'], bins=20)
        binned_mean = train_df.groupby(bins, observed=True)[sensor].mean()
        bin_centers = [(b.left + b.right) / 2 for b in binned_mean.index]
        ax.plot(bin_centers, binned_mean.values, color='red', linewidth=2, label='Binned mean')
        
        corr_val = train_df[sensor].corr(train_df['RUL'])
        ax.set_title(f'{sensor} (r={corr_val:.3f})', fontsize=10)
        ax.set_xlabel('RUL')
        ax.set_ylabel(sensor)
        ax.legend(fontsize=8)
    
    fig.suptitle(f'C-MAPSS {subset} - Sensor Values vs RUL\n(Top 6 sensors by correlation)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / f'{subset}_sensor_vs_rul.png')
    plt.close()
    print(f"  Sensor vs RUL plots saved")


def analyze_operating_conditions(train_df: pd.DataFrame, fig_dir: Path, 
                                   table_dir: Path, subset: str):
    """Analyze operating condition clusters."""
    op_cols = [c for c in train_df.columns if c.startswith('op_setting_')]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Scatter of operational settings
    ax = axes[0]
    ax.scatter(train_df['op_setting_1'], train_df['op_setting_2'], 
               alpha=0.05, s=2, c='steelblue')
    ax.set_xlabel('Op Setting 1')
    ax.set_ylabel('Op Setting 2')
    ax.set_title(f'{subset} - Operating Settings (1 vs 2)')
    
    ax = axes[1]
    ax.scatter(train_df['op_setting_1'], train_df['op_setting_3'], 
               alpha=0.05, s=2, c='steelblue')
    ax.set_xlabel('Op Setting 1')
    ax.set_ylabel('Op Setting 3')
    ax.set_title(f'{subset} - Operating Settings (1 vs 3)')
    
    ax = axes[2]
    ax.scatter(train_df['op_setting_2'], train_df['op_setting_3'], 
               alpha=0.05, s=2, c='steelblue')
    ax.set_xlabel('Op Setting 2')
    ax.set_ylabel('Op Setting 3')
    ax.set_title(f'{subset} - Operating Settings (2 vs 3)')
    
    fig.suptitle(f'C-MAPSS {subset} - Operating Condition Space', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / f'{subset}_operating_conditions.png')
    plt.close()
    
    # Count unique operating condition clusters
    # Round to identify clusters
    op_rounded = train_df[op_cols].round(2)
    n_clusters = op_rounded.drop_duplicates().shape[0]
    print(f"  Operating condition clusters (rounded): {n_clusters}")
    
    # Save operational setting statistics
    op_stats = train_df[op_cols].describe()
    op_stats.to_csv(table_dir / f'{subset}_operating_settings_stats.csv')
    print(f"  Operating conditions analysis saved")


def plot_all_subsets_comparison(data_dir: str, fig_dir: Path, table_dir: Path):
    """Compare all four subsets side by side."""
    summaries = {}
    for subset in ['FD001', 'FD002', 'FD003', 'FD004']:
        try:
            summaries[subset] = get_dataset_summary(data_dir, subset)
        except FileNotFoundError:
            continue
    
    if not summaries:
        return
    
    # Comparison table
    rows = []
    for name, s in summaries.items():
        rows.append({
            'Subset': name,
            'Op_Conditions': s['operating_conditions'],
            'Fault_Modes': s['fault_modes'],
            'Fault_Type': s['fault_type'],
            'Train_Engines': s['train']['n_engines'],
            'Test_Engines': s['test']['n_engines'],
            'Train_Obs': s['train']['n_observations'],
            'Test_Obs': s['test']['n_observations'],
            'Life_Min': s['train']['lifetime_min'],
            'Life_Max': s['train']['lifetime_max'],
            'Life_Mean': f"{s['train']['lifetime_mean']:.1f}",
            'Life_Std': f"{s['train']['lifetime_std']:.1f}",
            'Constant_Sensors': len(s['features']['constant_sensors']),
            'Informative_Sensors': len(s['features']['informative_sensors']),
            'Missing_Values': s['missing_values']['train'],
        })
    
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(table_dir / 'all_subsets_comparison.csv', index=False)
    print(f"\n  Subset comparison table saved")
    
    # Comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    subsets = list(summaries.keys())
    x = range(len(subsets))
    
    # Train engines
    train_engines = [summaries[s]['train']['n_engines'] for s in subsets]
    test_engines = [summaries[s]['test']['n_engines'] for s in subsets]
    
    ax = axes[0, 0]
    width = 0.35
    ax.bar([i - width/2 for i in x], train_engines, width, label='Train', color='steelblue')
    ax.bar([i + width/2 for i in x], test_engines, width, label='Test', color='coral')
    ax.set_xticks(list(x))
    ax.set_xticklabels(subsets)
    ax.set_ylabel('Number of Engines')
    ax.set_title('Engine Count by Subset')
    ax.legend()
    
    # Lifetimes
    ax = axes[0, 1]
    life_means = [summaries[s]['train']['lifetime_mean'] for s in subsets]
    life_stds = [summaries[s]['train']['lifetime_std'] for s in subsets]
    ax.bar(x, life_means, yerr=life_stds, color='steelblue', alpha=0.7, capsize=5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(subsets)
    ax.set_ylabel('Cycles')
    ax.set_title('Mean Engine Lifetime (+/- std)')
    
    # Observations
    ax = axes[1, 0]
    train_obs = [summaries[s]['train']['n_observations'] for s in subsets]
    test_obs = [summaries[s]['test']['n_observations'] for s in subsets]
    ax.bar([i - width/2 for i in x], train_obs, width, label='Train', color='steelblue')
    ax.bar([i + width/2 for i in x], test_obs, width, label='Test', color='coral')
    ax.set_xticks(list(x))
    ax.set_xticklabels(subsets)
    ax.set_ylabel('Observations')
    ax.set_title('Total Observations by Subset')
    ax.legend()
    
    # Complexity
    ax = axes[1, 1]
    op_conditions = [summaries[s]['operating_conditions'] for s in subsets]
    fault_modes = [summaries[s]['fault_modes'] for s in subsets]
    ax.bar([i - width/2 for i in x], op_conditions, width, label='Op. Conditions', color='steelblue')
    ax.bar([i + width/2 for i in x], fault_modes, width, label='Fault Modes', color='coral')
    ax.set_xticks(list(x))
    ax.set_xticklabels(subsets)
    ax.set_ylabel('Count')
    ax.set_title('Complexity by Subset')
    ax.legend()
    
    fig.suptitle('C-MAPSS Dataset Overview - All Subsets', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'all_subsets_comparison.png')
    plt.close()


def run_full_eda(data_dir: str = 'data/raw', subset: str = 'FD001'):
    """Run complete EDA for a single subset."""
    print(f"\n{'='*60}")
    print(f"Running EDA for C-MAPSS {subset}")
    print(f"{'='*60}")
    
    fig_dir = Path('results/figures/eda')
    table_dir = Path('results/tables/eda')
    ensure_dirs(fig_dir, table_dir)
    
    # Load data
    print("\n[1/9] Loading data...")
    train_df = load_train_data(data_dir, subset)
    test_df, rul_labels = load_test_data(data_dir, subset)
    print(f"  Train: {len(train_df)} rows, {train_df['unit_id'].nunique()} engines")
    print(f"  Test: {len(test_df)} rows, {test_df['unit_id'].nunique()} engines")
    
    # Feature dictionary
    print("\n[2/9] Building feature dictionary...")
    feat_df = analyze_feature_dictionary(train_df, table_dir, subset)
    
    # Missing value analysis
    print("\n[3/9] Missing value analysis...")
    train_missing = train_df.isnull().sum()
    test_missing = test_df.isnull().sum()
    print(f"  Train missing: {train_missing.sum()} total")
    print(f"  Test missing: {test_missing.sum()} total")
    
    # Constant feature analysis
    print("\n[4/9] Constant feature analysis...")
    constant, near_const, variable = analyze_constant_features(train_df, table_dir, subset)
    
    # Correlation analysis
    print("\n[5/9] Correlation analysis...")
    plot_correlation_matrix(train_df, variable, fig_dir, subset)
    
    # Sensor distributions
    print("\n[6/9] Sensor distributions...")
    plot_sensor_distributions(train_df, variable, fig_dir, subset)
    
    # Operating conditions
    print("\n[7/9] Operating condition analysis...")
    analyze_operating_conditions(train_df, fig_dir, table_dir, subset)
    
    # Engine lifetimes
    print("\n[8/9] Engine lifetime analysis...")
    plot_engine_lifetimes(train_df, fig_dir, subset)
    
    # Degradation trajectories
    print("\n[9/9] Degradation trajectory analysis...")
    plot_degradation_trajectories(train_df, variable, fig_dir, subset)
    plot_degradation_vs_rul(train_df, variable, fig_dir, subset)
    
    # Save summary JSON
    summary = get_dataset_summary(data_dir, subset)
    with open(table_dir / f'{subset}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"EDA complete for {subset}")
    print(f"Figures: {fig_dir}")
    print(f"Tables: {table_dir}")
    print(f"{'='*60}")
    
    return summary


def main():
    data_dir = 'data/raw'
    
    # Run EDA for each subset
    for subset in ['FD001', 'FD002', 'FD003', 'FD004']:
        run_full_eda(data_dir, subset)
    
    # Cross-subset comparison
    print("\n[CROSS-SUBSET] Generating comparison...")
    fig_dir = Path('results/figures/eda')
    table_dir = Path('results/tables/eda')
    plot_all_subsets_comparison(data_dir, fig_dir, table_dir)
    
    print("\n" + "=" * 60)
    print("FULL EDA COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
