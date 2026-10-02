"""NASA C-MAPSS dataset loader and inspector.

Loads raw C-MAPSS text files into structured DataFrames with proper column names.
Provides dataset statistics, feature dictionaries, and basic validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# C-MAPSS column definitions
COLUMN_NAMES = (
    ['unit_id', 'cycle'] +
    [f'op_setting_{i}' for i in range(1, 4)] +
    [f'sensor_{i}' for i in range(1, 22)]
)

# Sensor descriptions from C-MAPSS documentation
SENSOR_DESCRIPTIONS = {
    'sensor_1': 'Total temperature at fan inlet (degR)',
    'sensor_2': 'Total temperature at LPC outlet (degR)',
    'sensor_3': 'Total temperature at HPC outlet (degR)',
    'sensor_4': 'Total temperature at LPT outlet (degR)',
    'sensor_5': 'Pressure at fan inlet (psia)',
    'sensor_6': 'Total pressure in bypass-duct (psia)',
    'sensor_7': 'Total pressure at HPC outlet (psia)',
    'sensor_8': 'Physical fan speed (rpm)',
    'sensor_9': 'Physical core speed (rpm)',
    'sensor_10': 'Engine pressure ratio (P50/P2)',
    'sensor_11': 'Static pressure at HPC outlet (psia)',
    'sensor_12': 'Ratio of fuel flow to Ps30 (pps/psi)',
    'sensor_13': 'Corrected fan speed (rpm)',
    'sensor_14': 'Corrected core speed (rpm)',
    'sensor_15': 'Bypass ratio',
    'sensor_16': 'Burner fuel-air ratio',
    'sensor_17': 'Bleed enthalpy',
    'sensor_18': 'Demanded fan speed (rpm)',
    'sensor_19': 'Demanded corrected fan speed (rpm)',
    'sensor_20': 'HPT coolant bleed (lbm/s)',
    'sensor_21': 'LPT coolant bleed (lbm/s)',
}

OPERATIONAL_SETTING_DESCRIPTIONS = {
    'op_setting_1': 'Altitude (related)',
    'op_setting_2': 'Mach number (related)',
    'op_setting_3': 'Throttle resolver angle (related)',
}

# Dataset metadata
DATASET_INFO = {
    'FD001': {'op_conditions': 1, 'fault_modes': 1, 'fault_type': 'HPC degradation'},
    'FD002': {'op_conditions': 6, 'fault_modes': 1, 'fault_type': 'HPC degradation'},
    'FD003': {'op_conditions': 1, 'fault_modes': 2, 'fault_type': 'HPC + Fan degradation'},
    'FD004': {'op_conditions': 6, 'fault_modes': 2, 'fault_type': 'HPC + Fan degradation'},
}


def load_train_data(data_dir: str, subset: str = 'FD001') -> pd.DataFrame:
    """Load C-MAPSS training data.
    
    Args:
        data_dir: Path to directory containing raw data files.
        subset: Dataset subset ('FD001', 'FD002', 'FD003', 'FD004').
        
    Returns:
        DataFrame with named columns and computed RUL.
    """
    filepath = Path(data_dir) / f'train_{subset}.txt'
    if not filepath.exists():
        raise FileNotFoundError(f"Training data not found: {filepath}")
    
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=COLUMN_NAMES)
    
    # Compute RUL (Remaining Useful Life) for training data
    # RUL = max_cycle_for_engine - current_cycle
    max_cycles = df.groupby('unit_id')['cycle'].max()
    df['max_cycle'] = df['unit_id'].map(max_cycles)
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    
    return df


def load_test_data(data_dir: str, subset: str = 'FD001') -> Tuple[pd.DataFrame, pd.Series]:
    """Load C-MAPSS test data and RUL labels.
    
    Args:
        data_dir: Path to directory containing raw data files.
        subset: Dataset subset.
        
    Returns:
        Tuple of (test DataFrame, RUL labels Series).
    """
    test_path = Path(data_dir) / f'test_{subset}.txt'
    rul_path = Path(data_dir) / f'RUL_{subset}.txt'
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    if not rul_path.exists():
        raise FileNotFoundError(f"RUL file not found: {rul_path}")
    
    df = pd.read_csv(test_path, sep=r'\s+', header=None, names=COLUMN_NAMES)
    rul_labels = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
    
    # Compute RUL for test data
    # Test engines are cut off at some point before failure
    # RUL_label tells us how many cycles are left at the last observed cycle
    max_cycles = df.groupby('unit_id')['cycle'].max()
    
    rul_dict = {}
    for idx, (unit_id, max_cycle) in enumerate(max_cycles.items()):
        remaining = rul_labels.iloc[idx]['RUL']
        total_life = max_cycle + remaining
        rul_dict[unit_id] = total_life
    
    df['total_life'] = df['unit_id'].map(rul_dict)
    df['RUL'] = df['total_life'] - df['cycle']
    df.drop('total_life', axis=1, inplace=True)
    
    return df, rul_labels


def get_dataset_summary(data_dir: str, subset: str = 'FD001') -> Dict:
    """Generate a comprehensive summary of a C-MAPSS subset.
    
    Args:
        data_dir: Path to raw data directory.
        subset: Dataset subset.
        
    Returns:
        Dictionary containing dataset statistics.
    """
    train_df = load_train_data(data_dir, subset)
    test_df, rul_labels = load_test_data(data_dir, subset)
    
    info = DATASET_INFO[subset]
    
    sensor_cols = [c for c in COLUMN_NAMES if c.startswith('sensor_')]
    op_cols = [c for c in COLUMN_NAMES if c.startswith('op_setting_')]
    
    # Compute statistics
    train_engines = train_df['unit_id'].nunique()
    test_engines = test_df['unit_id'].nunique()
    
    train_lifetimes = train_df.groupby('unit_id')['cycle'].max()
    test_observed = test_df.groupby('unit_id')['cycle'].max()
    
    # Identify constant/near-constant sensors
    constant_sensors = []
    near_constant_sensors = []
    for col in sensor_cols:
        std = train_df[col].std()
        if std == 0:
            constant_sensors.append(col)
        elif std < 1e-6:
            near_constant_sensors.append(col)
    
    summary = {
        'subset': subset,
        'operating_conditions': info['op_conditions'],
        'fault_modes': info['fault_modes'],
        'fault_type': info['fault_type'],
        'train': {
            'n_engines': train_engines,
            'n_observations': len(train_df),
            'lifetime_min': int(train_lifetimes.min()),
            'lifetime_max': int(train_lifetimes.max()),
            'lifetime_mean': float(train_lifetimes.mean()),
            'lifetime_median': float(train_lifetimes.median()),
            'lifetime_std': float(train_lifetimes.std()),
        },
        'test': {
            'n_engines': test_engines,
            'n_observations': len(test_df),
            'observed_min': int(test_observed.min()),
            'observed_max': int(test_observed.max()),
            'rul_min': int(rul_labels['RUL'].min()),
            'rul_max': int(rul_labels['RUL'].max()),
            'rul_mean': float(rul_labels['RUL'].mean()),
        },
        'features': {
            'n_sensors': len(sensor_cols),
            'n_operational_settings': len(op_cols),
            'constant_sensors': constant_sensors,
            'near_constant_sensors': near_constant_sensors,
            'informative_sensors': [s for s in sensor_cols 
                                     if s not in constant_sensors 
                                     and s not in near_constant_sensors],
        },
        'missing_values': {
            'train': int(train_df.isnull().sum().sum()),
            'test': int(test_df.isnull().sum().sum()),
        },
    }
    
    return summary


def print_dataset_summary(summary: Dict) -> None:
    """Print a formatted dataset summary."""
    s = summary
    print(f"\n{'='*60}")
    print(f"Dataset: C-MAPSS {s['subset']}")
    print(f"{'='*60}")
    print(f"Operating Conditions: {s['operating_conditions']}")
    print(f"Fault Modes: {s['fault_modes']} ({s['fault_type']})")
    print(f"\n--- Training Data ---")
    print(f"Engines: {s['train']['n_engines']}")
    print(f"Total observations: {s['train']['n_observations']}")
    print(f"Lifetime range: {s['train']['lifetime_min']} - {s['train']['lifetime_max']} cycles")
    print(f"Mean lifetime: {s['train']['lifetime_mean']:.1f} +/- {s['train']['lifetime_std']:.1f} cycles")
    print(f"Median lifetime: {s['train']['lifetime_median']:.1f} cycles")
    print(f"\n--- Test Data ---")
    print(f"Engines: {s['test']['n_engines']}")
    print(f"Total observations: {s['test']['n_observations']}")
    print(f"Observed range: {s['test']['observed_min']} - {s['test']['observed_max']} cycles")
    print(f"RUL range: {s['test']['rul_min']} - {s['test']['rul_max']} cycles")
    print(f"Mean RUL at cutoff: {s['test']['rul_mean']:.1f} cycles")
    print(f"\n--- Features ---")
    print(f"Sensors: {s['features']['n_sensors']}")
    print(f"Operational settings: {s['features']['n_operational_settings']}")
    print(f"Constant sensors: {s['features']['constant_sensors']}")
    print(f"Informative sensors: {len(s['features']['informative_sensors'])}")
    print(f"\n--- Data Quality ---")
    print(f"Missing values (train): {s['missing_values']['train']}")
    print(f"Missing values (test): {s['missing_values']['test']}")


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/raw'
    
    for subset in ['FD001', 'FD002', 'FD003', 'FD004']:
        try:
            summary = get_dataset_summary(data_dir, subset)
            print_dataset_summary(summary)
        except FileNotFoundError as e:
            print(f"\nSkipping {subset}: {e}")
