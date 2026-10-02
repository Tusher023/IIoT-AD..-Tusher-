"""Normalization utilities for C-MAPSS data.

LEAKAGE PREVENTION:
- Scaler is ALWAYS fitted on training data only
- The same fitted scaler is applied to validation and test data
- Scaler parameters are saved for reproducibility
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class SafeScaler:
    """StandardScaler that enforces train-only fitting.
    
    Wraps sklearn-style normalization with explicit tracking of
    which data was used for fitting, preventing accidental leakage.
    """
    
    def __init__(self, method: str = 'standard'):
        """Initialize scaler.
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust').
        """
        if method not in ('standard', 'minmax', 'robust'):
            raise ValueError(f"Unknown method: {method}. Use 'standard', 'minmax', or 'robust'.")
        
        self.method = method
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.params: Dict = {}
    
    def fit(self, train_df: pd.DataFrame, feature_cols: List[str]) -> 'SafeScaler':
        """Fit scaler on training data ONLY.
        
        Args:
            train_df: Training DataFrame (must be train split only).
            feature_cols: Columns to normalize.
            
        Returns:
            Self for chaining.
        """
        self.feature_names = list(feature_cols)
        data = train_df[feature_cols]
        
        if self.method == 'standard':
            self.params = {
                'mean': data.mean().to_dict(),
                'std': data.std().to_dict(),
            }
            # Replace zero std with 1 to avoid division by zero
            for col in feature_cols:
                if self.params['std'][col] == 0 or np.isnan(self.params['std'][col]):
                    self.params['std'][col] = 1.0
                    
        elif self.method == 'minmax':
            self.params = {
                'min': data.min().to_dict(),
                'max': data.max().to_dict(),
            }
            # Replace zero range with 1
            for col in feature_cols:
                if self.params['max'][col] == self.params['min'][col]:
                    self.params['max'][col] = self.params['min'][col] + 1.0
                    
        elif self.method == 'robust':
            self.params = {
                'median': data.median().to_dict(),
                'iqr': (data.quantile(0.75) - data.quantile(0.25)).to_dict(),
            }
            for col in feature_cols:
                if self.params['iqr'][col] == 0 or np.isnan(self.params['iqr'][col]):
                    self.params['iqr'][col] = 1.0
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters.
        
        Args:
            df: DataFrame to transform (can be train, val, or test).
            
        Returns:
            Transformed DataFrame (copy, original is not modified).
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() with training data first.")
        
        result = df.copy()
        
        if self.method == 'standard':
            for col in self.feature_names:
                result[col] = (result[col] - self.params['mean'][col]) / self.params['std'][col]
                
        elif self.method == 'minmax':
            for col in self.feature_names:
                range_val = self.params['max'][col] - self.params['min'][col]
                result[col] = (result[col] - self.params['min'][col]) / range_val
                
        elif self.method == 'robust':
            for col in self.feature_names:
                result[col] = (result[col] - self.params['median'][col]) / self.params['iqr'][col]
        
        return result
    
    def fit_transform(self, train_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Fit on training data and transform it.
        
        Args:
            train_df: Training DataFrame.
            feature_cols: Columns to normalize.
            
        Returns:
            Transformed training DataFrame.
        """
        self.fit(train_df, feature_cols)
        return self.transform(train_df)
    
    def save(self, path: str) -> None:
        """Save scaler parameters for reproducibility.
        
        Args:
            path: Output file path (JSON).
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'method': self.method,
            'feature_names': self.feature_names,
            'params': self.params,
            'is_fitted': self.is_fitted,
        }
        
        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SafeScaler':
        """Load a previously saved scaler.
        
        Args:
            path: Path to saved scaler JSON.
            
        Returns:
            Loaded SafeScaler instance.
        """
        with open(path, 'r') as f:
            state = json.load(f)
        
        scaler = cls(method=state['method'])
        scaler.feature_names = state['feature_names']
        scaler.params = state['params']
        scaler.is_fitted = state['is_fitted']
        return scaler


def get_feature_columns(
    df: pd.DataFrame,
    include_sensors: bool = True,
    include_operational: bool = True,
    exclude_constant: bool = True,
    constant_threshold: float = 0.0
) -> List[str]:
    """Get list of feature columns for modeling.
    
    Args:
        df: DataFrame to inspect.
        include_sensors: Include sensor_* columns.
        include_operational: Include op_setting_* columns.
        exclude_constant: Remove constant features (std=0).
        constant_threshold: Std threshold below which a feature is constant.
        
    Returns:
        List of feature column names.
    """
    cols = []
    
    if include_sensors:
        cols.extend([c for c in df.columns if c.startswith('sensor_')])
    if include_operational:
        cols.extend([c for c in df.columns if c.startswith('op_setting_')])
    
    if exclude_constant:
        non_constant = []
        for col in cols:
            if df[col].std() > constant_threshold:
                non_constant.append(col)
        cols = non_constant
    
    return sorted(cols)
