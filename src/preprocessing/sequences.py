"""Sequence generation for LSTM-based models.

Generates sliding-window sequences from engine time-series data.

LEAKAGE PREVENTION:
- Sequences NEVER cross engine boundaries
- Sequences NEVER use future information
- Each sequence contains only past/current timesteps
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset


def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int = 30,
    stride: int = 1,
    unit_col: str = 'unit_id',
    cycle_col: str = 'cycle'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding-window sequences from engine data.
    
    Each sequence contains `sequence_length` consecutive timesteps from
    a SINGLE engine. Sequences never cross engine boundaries.
    
    Args:
        df: DataFrame with features sorted by (unit_id, cycle).
        feature_cols: Columns to include in sequences.
        sequence_length: Number of timesteps per sequence.
        stride: Step size between consecutive sequences.
        unit_col: Engine identifier column.
        cycle_col: Time column.
        
    Returns:
        Tuple of:
            sequences: np.ndarray of shape (N, sequence_length, n_features)
            engine_ids: np.ndarray of shape (N,) - engine ID for each sequence
            end_cycles: np.ndarray of shape (N,) - last cycle in each sequence
    """
    sequences = []
    engine_ids = []
    end_cycles = []
    
    # Sort by engine and cycle to ensure temporal ordering
    df_sorted = df.sort_values([unit_col, cycle_col])
    
    for engine_id, engine_data in df_sorted.groupby(unit_col):
        values = engine_data[feature_cols].values
        cycles = engine_data[cycle_col].values
        n_timesteps = len(values)
        
        if n_timesteps < sequence_length:
            continue  # Skip engines with too few timesteps
        
        # Sliding window within this engine only
        for start in range(0, n_timesteps - sequence_length + 1, stride):
            end = start + sequence_length
            seq = values[start:end]
            sequences.append(seq)
            engine_ids.append(engine_id)
            end_cycles.append(cycles[end - 1])
    
    if not sequences:
        raise ValueError(
            f"No sequences generated. Check sequence_length ({sequence_length}) "
            f"vs min engine length."
        )
    
    return (
        np.array(sequences, dtype=np.float32),
        np.array(engine_ids),
        np.array(end_cycles)
    )


def create_sequences_with_rul(
    df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int = 30,
    stride: int = 1,
    unit_col: str = 'unit_id',
    cycle_col: str = 'cycle',
    rul_col: str = 'RUL'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sequences with associated RUL values.
    
    Same as create_sequences but also returns the RUL at the end of
    each sequence (used for evaluation, NOT for training unsupervised models).
    
    Returns:
        Tuple of (sequences, engine_ids, end_cycles, rul_values)
    """
    sequences = []
    engine_ids_list = []
    end_cycles = []
    rul_values = []
    
    df_sorted = df.sort_values([unit_col, cycle_col])
    
    for engine_id, engine_data in df_sorted.groupby(unit_col):
        values = engine_data[feature_cols].values
        cycles = engine_data[cycle_col].values
        ruls = engine_data[rul_col].values
        n_timesteps = len(values)
        
        if n_timesteps < sequence_length:
            continue
        
        for start in range(0, n_timesteps - sequence_length + 1, stride):
            end = start + sequence_length
            sequences.append(values[start:end])
            engine_ids_list.append(engine_id)
            end_cycles.append(cycles[end - 1])
            rul_values.append(ruls[end - 1])
    
    return (
        np.array(sequences, dtype=np.float32),
        np.array(engine_ids_list),
        np.array(end_cycles),
        np.array(rul_values, dtype=np.float32)
    )


def extract_normal_data(
    df: pd.DataFrame,
    normal_ratio: float = 0.7,
    unit_col: str = 'unit_id',
    cycle_col: str = 'cycle'
) -> pd.DataFrame:
    """Extract the "normal" operating period from each engine.
    
    The first `normal_ratio` fraction of each engine's life is considered
    normal (healthy) operation. This is used for training autoencoders
    on normal data only.
    
    Args:
        df: DataFrame with engine trajectories.
        normal_ratio: Fraction of each engine's life considered normal.
        unit_col: Engine identifier column.
        cycle_col: Time column.
        
    Returns:
        DataFrame containing only the normal operating period.
    """
    normal_dfs = []
    
    for engine_id, engine_data in df.groupby(unit_col):
        engine_sorted = engine_data.sort_values(cycle_col)
        total_cycles = len(engine_sorted)
        n_normal = max(1, int(total_cycles * normal_ratio))
        
        normal_dfs.append(engine_sorted.iloc[:n_normal])
    
    return pd.concat(normal_dfs, ignore_index=True)


class SequenceDataset(Dataset):
    """PyTorch Dataset for LSTM Autoencoder sequences."""
    
    def __init__(self, sequences: np.ndarray):
        """Initialize dataset.
        
        Args:
            sequences: Array of shape (N, seq_len, n_features).
        """
        self.sequences = torch.FloatTensor(sequences)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


class TabularDataset(Dataset):
    """PyTorch Dataset for feedforward Autoencoder (per-timestep)."""
    
    def __init__(self, data: np.ndarray):
        """Initialize dataset.
        
        Args:
            data: Array of shape (N, n_features).
        """
        self.data = torch.FloatTensor(data)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
