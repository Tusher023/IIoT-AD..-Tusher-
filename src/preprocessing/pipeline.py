"""Data preprocessing pipeline orchestrator.

Combines loading, splitting, normalization, and sequence generation
into a single configurable pipeline with full leakage prevention.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from src.data.cmapss_loader import load_train_data, COLUMN_NAMES
from src.preprocessing.splitting import engine_level_split, verify_no_leakage, save_split_info
from src.preprocessing.normalization import SafeScaler, get_feature_columns
from src.preprocessing.sequences import (
    create_sequences, create_sequences_with_rul,
    extract_normal_data, SequenceDataset, TabularDataset
)


class DataPipeline:
    """End-to-end data preprocessing pipeline with leakage prevention.
    
    Usage:
        pipeline = DataPipeline(config)
        data = pipeline.prepare()
        # data contains train/val/test splits, normalized, with sequences
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline from configuration.
        
        Args:
            config: Configuration dictionary (from YAML).
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.feature_config = config.get('features', {})
        
        self.dataset = self.data_config.get('dataset', 'FD001')
        self.data_dir = self.data_config.get('data_dir', 'data/raw')
        self.output_dir = self.data_config.get('output_dir', 'data/processed')
        self.seed = config.get('experiment', {}).get('seed', 42)
        
        self.scaler: Optional[SafeScaler] = None
        self.feature_cols: List[str] = []
        self.split_info: Dict = {}
    
    def prepare(self, verbose: bool = True) -> Dict[str, Any]:
        """Run the full preprocessing pipeline.
        
        Returns:
            Dictionary containing all preprocessed data:
            {
                'train_df': pd.DataFrame,
                'val_df': pd.DataFrame,
                'test_df': pd.DataFrame,
                'feature_cols': List[str],
                'scaler': SafeScaler,
                'split_info': Dict,
                'train_normal_df': pd.DataFrame,  # Normal-period training data
            }
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Data Pipeline: C-MAPSS {self.dataset}")
            print(f"{'='*60}")
        
        # Step 1: Load data
        if verbose:
            print("\n[1/5] Loading data...")
        full_df = load_train_data(self.data_dir, self.dataset)
        if verbose:
            print(f"  Loaded {len(full_df)} observations, "
                  f"{full_df['unit_id'].nunique()} engines")
        
        # Step 2: Determine feature columns
        if verbose:
            print("\n[2/5] Selecting features...")
        self.feature_cols = self._select_features(full_df)
        if verbose:
            print(f"  Selected {len(self.feature_cols)} features: {self.feature_cols}")
        
        # Step 3: Engine-level split
        if verbose:
            print("\n[3/5] Engine-level splitting...")
        train_df, val_df, test_df, self.split_info = engine_level_split(
            full_df,
            train_ratio=self.data_config.get('train_ratio', 0.70),
            val_ratio=self.data_config.get('val_ratio', 0.15),
            test_ratio=self.data_config.get('test_ratio', 0.15),
            seed=self.seed
        )
        verify_no_leakage(train_df, val_df, test_df)
        if verbose:
            print(f"  Train: {len(train_df)} obs ({self.split_info['n_train_engines']} engines)")
            print(f"  Val:   {len(val_df)} obs ({self.split_info['n_val_engines']} engines)")
            print(f"  Test:  {len(test_df)} obs ({self.split_info['n_test_engines']} engines)")
        
        # Step 4: Normalize using TRAIN statistics only
        if verbose:
            print("\n[4/5] Normalizing (train-only statistics)...")
        norm_method = self.data_config.get('normalization', 'standard')
        self.scaler = SafeScaler(method=norm_method)
        
        train_df = self.scaler.fit_transform(train_df, self.feature_cols)
        val_df = self.scaler.transform(val_df)
        test_df = self.scaler.transform(test_df)
        if verbose:
            print(f"  Method: {norm_method}")
            print(f"  Fitted on {self.split_info['n_train_engines']} training engines only")
        
        # Step 5: Extract normal training data
        if verbose:
            print("\n[5/5] Extracting normal operating period...")
        normal_ratio = self.config.get('training', {}).get('normal_ratio', 0.7)
        train_normal_df = extract_normal_data(train_df, normal_ratio=normal_ratio)
        if verbose:
            print(f"  Normal ratio: {normal_ratio}")
            print(f"  Normal training observations: {len(train_normal_df)} "
                  f"(of {len(train_df)} total)")
        
        # Save artifacts
        self._save_artifacts(train_df, val_df, test_df)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Pipeline complete. No leakage detected.")
            print(f"{'='*60}")
        
        return {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'train_normal_df': train_normal_df,
            'feature_cols': self.feature_cols,
            'scaler': self.scaler,
            'split_info': self.split_info,
        }
    
    def prepare_sequences(
        self,
        data: Dict[str, Any],
        sequence_length: Optional[int] = None,
        stride: int = 1
    ) -> Dict[str, Any]:
        """Generate sequences for LSTM models from preprocessed data.
        
        Args:
            data: Output from prepare().
            sequence_length: Window size (overrides config if given).
            stride: Sliding window stride.
            
        Returns:
            Dictionary with sequence arrays and metadata.
        """
        seq_len = sequence_length or self.data_config.get('sequence_length', 30)
        feature_cols = data['feature_cols']
        
        print(f"\nGenerating sequences (length={seq_len}, stride={stride})...")
        
        # Normal training sequences
        train_seqs, train_eids, train_cycles = create_sequences(
            data['train_normal_df'], feature_cols,
            sequence_length=seq_len, stride=stride
        )
        print(f"  Train (normal): {train_seqs.shape}")
        
        # Validation sequences (full trajectory, with RUL)
        val_seqs, val_eids, val_cycles, val_ruls = create_sequences_with_rul(
            data['val_df'], feature_cols,
            sequence_length=seq_len, stride=stride
        )
        print(f"  Val: {val_seqs.shape}")
        
        # Test sequences (full trajectory, with RUL)
        test_seqs, test_eids, test_cycles, test_ruls = create_sequences_with_rul(
            data['test_df'], feature_cols,
            sequence_length=seq_len, stride=stride
        )
        print(f"  Test: {test_seqs.shape}")
        
        return {
            'train_sequences': train_seqs,
            'train_engine_ids': train_eids,
            'train_cycles': train_cycles,
            'val_sequences': val_seqs,
            'val_engine_ids': val_eids,
            'val_cycles': val_cycles,
            'val_ruls': val_ruls,
            'test_sequences': test_seqs,
            'test_engine_ids': test_eids,
            'test_cycles': test_cycles,
            'test_ruls': test_ruls,
            'sequence_length': seq_len,
            'n_features': train_seqs.shape[2],
        }
    
    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select feature columns based on configuration."""
        selected = self.feature_config.get('selected_sensors', None)
        include_op = self.feature_config.get('include_operational_settings', True)
        
        if selected is not None:
            # Use explicitly specified sensors
            cols = [f'sensor_{i}' for i in selected]
            if include_op:
                op_cols = [c for c in df.columns if c.startswith('op_setting_')]
                cols.extend(op_cols)
            return sorted(cols)
        
        # Auto-select: all non-constant features
        return get_feature_columns(
            df,
            include_sensors=True,
            include_operational=include_op,
            exclude_constant=True
        )
    
    def _save_artifacts(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """Save pipeline artifacts for reproducibility."""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save split info
        save_split_info(
            self.split_info,
            str(output_path / f'{self.dataset}_split_info.json')
        )
        
        # Save scaler
        self.scaler.save(str(output_path / f'{self.dataset}_scaler.json'))
        
        # Save feature list
        with open(output_path / f'{self.dataset}_features.json', 'w') as f:
            json.dump({'feature_cols': self.feature_cols}, f, indent=2)
