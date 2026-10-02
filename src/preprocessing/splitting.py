"""Engine-level data splitting for C-MAPSS datasets.

Implements strict engine-level train/validation/test splits to prevent
temporal data leakage. All splits are deterministic given a random seed.

LEAKAGE PREVENTION:
- Splits are performed at the ENGINE level, never observation level
- No engine appears in multiple splits
- Split assignments are logged for reproducibility
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


def engine_level_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    unit_col: str = 'unit_id'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Split data by engine ID to prevent temporal leakage.
    
    Each engine's ENTIRE trajectory goes to exactly one split.
    No engine appears in more than one split.
    
    Args:
        df: Full dataset with unit_id column.
        train_ratio: Fraction of engines for training.
        val_ratio: Fraction of engines for validation.
        test_ratio: Fraction of engines for testing.
        seed: Random seed for reproducibility.
        unit_col: Column name for engine/unit identifier.
        
    Returns:
        Tuple of (train_df, val_df, test_df, split_info dict).
        
    Raises:
        ValueError: If ratios don't sum to ~1.0 or data is invalid.
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.3f}")
    
    # Get unique engine IDs
    engine_ids = np.array(sorted(df[unit_col].unique()))
    n_engines = len(engine_ids)
    
    if n_engines < 3:
        raise ValueError(f"Need at least 3 engines for 3-way split, got {n_engines}")
    
    # Shuffle deterministically
    rng = np.random.RandomState(seed)
    shuffled_ids = engine_ids.copy()
    rng.shuffle(shuffled_ids)
    
    # Compute split boundaries
    n_train = max(1, int(n_engines * train_ratio))
    n_val = max(1, int(n_engines * val_ratio))
    n_test = n_engines - n_train - n_val  # Remainder goes to test
    
    if n_test < 1:
        n_test = 1
        n_train = n_engines - n_val - n_test
    
    # Assign engines to splits
    train_ids = sorted(shuffled_ids[:n_train].tolist())
    val_ids = sorted(shuffled_ids[n_train:n_train + n_val].tolist())
    test_ids = sorted(shuffled_ids[n_train + n_val:].tolist())
    
    # Verify no overlap
    assert len(set(train_ids) & set(val_ids)) == 0, "Train/Val overlap!"
    assert len(set(train_ids) & set(test_ids)) == 0, "Train/Test overlap!"
    assert len(set(val_ids) & set(test_ids)) == 0, "Val/Test overlap!"
    assert len(train_ids) + len(val_ids) + len(test_ids) == n_engines, "Engine count mismatch!"
    
    # Split the data
    train_df = df[df[unit_col].isin(train_ids)].copy()
    val_df = df[df[unit_col].isin(val_ids)].copy()
    test_df = df[df[unit_col].isin(test_ids)].copy()
    
    # Build split info for logging/reproducibility
    split_info = {
        'seed': seed,
        'n_engines_total': n_engines,
        'n_train_engines': len(train_ids),
        'n_val_engines': len(val_ids),
        'n_test_engines': len(test_ids),
        'train_engine_ids': train_ids,
        'val_engine_ids': val_ids,
        'test_engine_ids': test_ids,
        'train_observations': len(train_df),
        'val_observations': len(val_df),
        'test_observations': len(test_df),
        'ratios_actual': {
            'train': len(train_ids) / n_engines,
            'val': len(val_ids) / n_engines,
            'test': len(test_ids) / n_engines,
        }
    }
    
    return train_df, val_df, test_df, split_info


def save_split_info(split_info: Dict, output_path: str) -> None:
    """Save split information for reproducibility.
    
    Args:
        split_info: Dictionary from engine_level_split.
        output_path: Path to save JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(split_info, f, indent=2)


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    unit_col: str = 'unit_id'
) -> bool:
    """Verify that no engine appears in multiple splits.
    
    Args:
        train_df, val_df, test_df: Split DataFrames.
        unit_col: Column name for engine identifier.
        
    Returns:
        True if no leakage detected.
        
    Raises:
        AssertionError: If leakage is detected.
    """
    train_ids = set(train_df[unit_col].unique())
    val_ids = set(val_df[unit_col].unique())
    test_ids = set(test_df[unit_col].unique())
    
    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids
    
    assert len(overlap_tv) == 0, f"LEAKAGE: Train/Val share engines: {overlap_tv}"
    assert len(overlap_tt) == 0, f"LEAKAGE: Train/Test share engines: {overlap_tt}"
    assert len(overlap_vt) == 0, f"LEAKAGE: Val/Test share engines: {overlap_vt}"
    
    print(f"  Leakage check PASSED: Train({len(train_ids)}) | "
          f"Val({len(val_ids)}) | Test({len(test_ids)}) engines, no overlap")
    return True
