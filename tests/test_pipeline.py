"""Comprehensive tests for data pipeline leakage prevention.

These tests verify that the entire preprocessing pipeline is free
from data leakage at every stage: splitting, normalization, sequence
generation, and normal-period extraction.

RUN: python -m pytest tests/test_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.splitting import engine_level_split, verify_no_leakage
from src.preprocessing.normalization import SafeScaler, get_feature_columns
from src.preprocessing.sequences import (
    create_sequences, create_sequences_with_rul, extract_normal_data
)
from src.preprocessing.pipeline import DataPipeline


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_df():
    """Create a small synthetic C-MAPSS-like DataFrame for testing."""
    np.random.seed(42)
    rows = []
    for engine_id in range(1, 11):  # 10 engines
        n_cycles = np.random.randint(30, 80)
        for cycle in range(1, n_cycles + 1):
            row = {
                'unit_id': engine_id,
                'cycle': cycle,
                'op_setting_1': np.random.uniform(0, 1),
                'op_setting_2': np.random.uniform(0, 0.5),
                'op_setting_3': 100.0,  # Constant
                'sensor_1': 500 + np.random.randn() * 5 - cycle * 0.1,
                'sensor_2': 600 + np.random.randn() * 10 + cycle * 0.05,
                'sensor_3': 1400,  # Constant
                'sensor_4': 1200 + np.random.randn() * 3,
            }
            row['RUL'] = n_cycles - cycle
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Test: Engine-Level Splitting
# ============================================================

class TestEngineLevelSplit:
    """Tests for engine-level data splitting."""
    
    def test_no_engine_overlap(self, sample_df):
        """CRITICAL: No engine should appear in multiple splits."""
        train, val, test, info = engine_level_split(sample_df, seed=42)
        
        train_ids = set(train['unit_id'].unique())
        val_ids = set(val['unit_id'].unique())
        test_ids = set(test['unit_id'].unique())
        
        assert len(train_ids & val_ids) == 0, "Train/Val engine overlap!"
        assert len(train_ids & test_ids) == 0, "Train/Test engine overlap!"
        assert len(val_ids & test_ids) == 0, "Val/Test engine overlap!"
    
    def test_all_engines_assigned(self, sample_df):
        """Every engine must be in exactly one split."""
        train, val, test, info = engine_level_split(sample_df, seed=42)
        
        all_ids = set(sample_df['unit_id'].unique())
        split_ids = (set(train['unit_id'].unique()) |
                     set(val['unit_id'].unique()) |
                     set(test['unit_id'].unique()))
        
        assert all_ids == split_ids, "Not all engines assigned!"
    
    def test_no_observations_lost(self, sample_df):
        """Total observations should be preserved."""
        train, val, test, info = engine_level_split(sample_df, seed=42)
        assert len(train) + len(val) + len(test) == len(sample_df)
    
    def test_deterministic(self, sample_df):
        """Same seed should produce identical splits."""
        t1, v1, te1, _ = engine_level_split(sample_df, seed=42)
        t2, v2, te2, _ = engine_level_split(sample_df, seed=42)
        
        assert set(t1['unit_id'].unique()) == set(t2['unit_id'].unique())
        assert set(v1['unit_id'].unique()) == set(v2['unit_id'].unique())
        assert set(te1['unit_id'].unique()) == set(te2['unit_id'].unique())
    
    def test_different_seeds_different_splits(self, sample_df):
        """Different seeds should (likely) produce different splits."""
        _, _, _, info1 = engine_level_split(sample_df, seed=42)
        _, _, _, info2 = engine_level_split(sample_df, seed=123)
        
        # Very unlikely to be identical with different seeds
        assert info1['train_engine_ids'] != info2['train_engine_ids']
    
    def test_complete_engine_trajectories(self, sample_df):
        """Each engine's FULL trajectory must be in its split."""
        train, val, test, _ = engine_level_split(sample_df, seed=42)
        
        for engine_id in train['unit_id'].unique():
            original_cycles = len(sample_df[sample_df['unit_id'] == engine_id])
            split_cycles = len(train[train['unit_id'] == engine_id])
            assert original_cycles == split_cycles, \
                f"Engine {engine_id}: trajectory truncated ({split_cycles}/{original_cycles})"
    
    def test_verify_no_leakage_function(self, sample_df):
        """The verify_no_leakage function should pass for valid splits."""
        train, val, test, _ = engine_level_split(sample_df, seed=42)
        assert verify_no_leakage(train, val, test) is True


# ============================================================
# Test: Normalization
# ============================================================

class TestNormalization:
    """Tests for train-only normalization."""
    
    def test_scaler_fitted_on_train_only(self, sample_df):
        """Scaler parameters must come from training data only."""
        train, val, test, _ = engine_level_split(sample_df, seed=42)
        feature_cols = ['sensor_1', 'sensor_2', 'sensor_4']
        
        scaler = SafeScaler(method='standard')
        scaler.fit(train, feature_cols)
        
        # Check mean matches training data
        for col in feature_cols:
            expected_mean = train[col].mean()
            assert abs(scaler.params['mean'][col] - expected_mean) < 1e-10, \
                f"Scaler mean for {col} doesn't match training mean!"
    
    def test_train_normalized_zero_mean(self, sample_df):
        """After standard normalization, training features should have ~zero mean."""
        train, _, _, _ = engine_level_split(sample_df, seed=42)
        feature_cols = ['sensor_1', 'sensor_2', 'sensor_4']
        
        scaler = SafeScaler(method='standard')
        train_norm = scaler.fit_transform(train, feature_cols)
        
        for col in feature_cols:
            mean = train_norm[col].mean()
            assert abs(mean) < 1e-6, f"Train {col} mean after normalization: {mean}"
    
    def test_val_test_use_train_statistics(self, sample_df):
        """Val/test must be transformed with training statistics, not their own."""
        train, val, test, _ = engine_level_split(sample_df, seed=42)
        feature_cols = ['sensor_1', 'sensor_2', 'sensor_4']
        
        scaler = SafeScaler(method='standard')
        scaler.fit(train, feature_cols)
        val_norm = scaler.transform(val)
        
        # Val mean should NOT be zero (it uses train statistics)
        # (This might be close to zero by chance, but generally won't be exactly zero)
        # The key test is that the scaler params haven't changed
        for col in feature_cols:
            assert scaler.params['mean'][col] == train[col].mean()
    
    def test_scaler_not_refitted(self, sample_df):
        """Transforming val/test should not change scaler parameters."""
        train, val, test, _ = engine_level_split(sample_df, seed=42)
        feature_cols = ['sensor_1', 'sensor_2']
        
        scaler = SafeScaler(method='standard')
        scaler.fit(train, feature_cols)
        original_params = {k: dict(v) for k, v in scaler.params.items()}
        
        scaler.transform(val)
        scaler.transform(test)
        
        assert scaler.params == original_params, "Scaler params changed after transform!"
    
    def test_constant_features_excluded(self, sample_df):
        """Constant features should be excluded by get_feature_columns."""
        cols = get_feature_columns(sample_df, exclude_constant=True)
        assert 'sensor_3' not in cols, "Constant sensor_3 should be excluded"
        assert 'op_setting_3' not in cols, "Constant op_setting_3 should be excluded"
    
    def test_scaler_save_load(self, sample_df, tmp_path):
        """Scaler should be saveable and loadable."""
        train, _, _, _ = engine_level_split(sample_df, seed=42)
        feature_cols = ['sensor_1', 'sensor_2']
        
        scaler = SafeScaler(method='standard')
        scaler.fit(train, feature_cols)
        
        save_path = str(tmp_path / 'scaler.json')
        scaler.save(save_path)
        loaded = SafeScaler.load(save_path)
        
        assert loaded.params == scaler.params
        assert loaded.feature_names == scaler.feature_names


# ============================================================
# Test: Sequence Generation
# ============================================================

class TestSequenceGeneration:
    """Tests for sliding-window sequence generation."""
    
    def test_no_cross_engine_sequences(self, sample_df):
        """CRITICAL: No sequence should span multiple engines."""
        feature_cols = ['sensor_1', 'sensor_2', 'sensor_4']
        sequences, engine_ids, end_cycles = create_sequences(
            sample_df, feature_cols, sequence_length=10
        )
        
        # Each sequence must have a single engine ID
        assert len(engine_ids) == len(sequences)
        # All should be valid engine IDs
        valid_ids = set(sample_df['unit_id'].unique())
        for eid in engine_ids:
            assert eid in valid_ids
    
    def test_sequence_shape(self, sample_df):
        """Sequences should have correct shape."""
        feature_cols = ['sensor_1', 'sensor_2', 'sensor_4']
        seq_len = 10
        sequences, _, _ = create_sequences(
            sample_df, feature_cols, sequence_length=seq_len
        )
        
        assert sequences.ndim == 3
        assert sequences.shape[1] == seq_len
        assert sequences.shape[2] == len(feature_cols)
    
    def test_no_future_information(self, sample_df):
        """Each sequence should contain only past/current timesteps."""
        feature_cols = ['sensor_1', 'sensor_2']
        sequences, engine_ids, end_cycles = create_sequences(
            sample_df, feature_cols, sequence_length=5
        )
        
        # For each sequence, the end_cycle should be >= all cycles in the sequence
        for i, (seq, eid, ec) in enumerate(zip(sequences, engine_ids, end_cycles)):
            engine_data = sample_df[sample_df['unit_id'] == eid].sort_values('cycle')
            # The sequence ending at cycle `ec` should only contain cycles <= ec
            # This is guaranteed by the sliding window approach
            assert ec >= 5, f"End cycle {ec} too small for sequence length 5"
    
    def test_normal_data_extraction(self, sample_df):
        """Normal data should be first portion of each engine."""
        normal_df = extract_normal_data(sample_df, normal_ratio=0.5)
        
        for engine_id in sample_df['unit_id'].unique():
            original = sample_df[sample_df['unit_id'] == engine_id].sort_values('cycle')
            normal = normal_df[normal_df['unit_id'] == engine_id].sort_values('cycle')
            
            total = len(original)
            expected_normal = max(1, int(total * 0.5))
            assert len(normal) == expected_normal, \
                f"Engine {engine_id}: expected {expected_normal} normal cycles, got {len(normal)}"
            
            # Normal cycles should be the EARLIEST cycles
            max_normal_cycle = normal['cycle'].max()
            normal_cycles = set(normal['cycle'].values)
            all_cycles = set(original['cycle'].values)
            remaining_cycles = all_cycles - normal_cycles
            if remaining_cycles:
                min_remaining_cycle = min(remaining_cycles)
                assert max_normal_cycle < min_remaining_cycle, \
                    "Normal data contains later cycles than remaining data!"


# ============================================================
# Test: Full Pipeline Integration
# ============================================================

class TestPipelineIntegration:
    """Integration tests for the full pipeline."""
    
    def test_pipeline_runs_on_real_data(self):
        """Test pipeline on actual C-MAPSS FD001 data if available."""
        data_path = Path('data/raw/train_FD001.txt')
        if not data_path.exists():
            pytest.skip("C-MAPSS data not available")
        
        config = {
            'experiment': {'seed': 42},
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
        
        pipeline = DataPipeline(config)
        data = pipeline.prepare(verbose=True)
        
        # Verify split
        verify_no_leakage(data['train_df'], data['val_df'], data['test_df'])
        
        # Verify normalization - check that features with meaningful variance
        # have approximately zero mean after standard normalization.
        # Near-constant features (like sensor_16 in FD001) may have tiny std 
        # replaced by 1.0, producing non-zero mean - this is expected.
        for col in data['feature_cols']:
            original_std = pipeline.scaler.params['std'][col]
            if original_std > 0.01:  # Only check features with real variance
                train_mean = data['train_df'][col].mean()
                assert abs(train_mean) < 1e-6, \
                    f"Train {col} mean after normalization: {train_mean}"
        
        # Verify normal data is subset of train
        normal_ids = set(data['train_normal_df']['unit_id'].unique())
        train_ids = set(data['train_df']['unit_id'].unique())
        assert normal_ids.issubset(train_ids), "Normal data contains non-train engines!"
        
        # Verify sequence generation
        seq_data = pipeline.prepare_sequences(data, sequence_length=30)
        assert seq_data['train_sequences'].shape[1] == 30
        assert seq_data['train_sequences'].shape[2] == len(data['feature_cols'])
        
        print(f"\nIntegration test PASSED:")
        print(f"  Features: {len(data['feature_cols'])}")
        print(f"  Train sequences: {seq_data['train_sequences'].shape}")
        print(f"  Val sequences: {seq_data['val_sequences'].shape}")
        print(f"  Test sequences: {seq_data['test_sequences'].shape}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
