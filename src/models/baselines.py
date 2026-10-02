"""Baseline anomaly detection models.

Implements three baseline methods:
1. Statistical Threshold (z-score based)
2. Isolation Forest
3. One-Class SVM

All models follow a consistent interface:
- fit(normal_data) — train on normal/healthy data
- score(data) — return anomaly scores (higher = more anomalous)
- get_params() — return model parameters for logging
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class StatisticalThresholdDetector:
    """Z-score based anomaly detector.
    
    Computes the mean and std of each feature from normal data,
    then scores new observations by their aggregate z-score.
    
    This is the simplest possible baseline. If complex models
    can't beat this convincingly, their added complexity is questionable.
    """
    
    def __init__(self, aggregation: str = 'mean'):
        """Initialize detector.
        
        Args:
            aggregation: How to aggregate per-feature z-scores.
                'mean': Mean absolute z-score across features.
                'max': Maximum absolute z-score across features.
                'sum': Sum of squared z-scores (Mahalanobis-like).
        """
        if aggregation not in ('mean', 'max', 'sum'):
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        self.aggregation = aggregation
        self.mean_ = None
        self.std_ = None
        self.fit_time_ = 0.0
        self.n_features_ = 0
    
    def fit(self, X: np.ndarray) -> 'StatisticalThresholdDetector':
        """Fit on normal/healthy data.
        
        Args:
            X: Normal data array of shape (n_samples, n_features).
            
        Returns:
            Self.
        """
        start = time.time()
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Replace zero std with 1 to avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        self.n_features_ = X.shape[1]
        self.fit_time_ = time.time() - start
        return self
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores.
        
        Args:
            X: Data array of shape (n_samples, n_features).
            
        Returns:
            Anomaly scores of shape (n_samples,). Higher = more anomalous.
        """
        z_scores = np.abs((X - self.mean_) / self.std_)
        
        if self.aggregation == 'mean':
            return np.mean(z_scores, axis=1)
        elif self.aggregation == 'max':
            return np.max(z_scores, axis=1)
        elif self.aggregation == 'sum':
            return np.sum(z_scores ** 2, axis=1)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters for logging."""
        return {
            'model_type': 'StatisticalThreshold',
            'aggregation': self.aggregation,
            'n_features': self.n_features_,
            'fit_time_seconds': self.fit_time_,
            'n_parameters': self.n_features_ * 2,
        }


class IsolationForestDetector:
    """Isolation Forest anomaly detector."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        contamination: str = 'auto',
        max_samples: str = 'auto',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.fit_time_ = 0.0
        self.n_estimators = n_estimators
        self.random_state = random_state
    
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        start = time.time()
        self.model.fit(X)
        self.fit_time_ = time.time() - start
        return self
    
    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'model_type': 'IsolationForest',
            'n_estimators': self.n_estimators,
            'contamination': str(self.model.contamination),
            'max_samples': str(self.model.max_samples),
            'random_state': self.random_state,
            'fit_time_seconds': self.fit_time_,
            'n_parameters': 'N/A (tree ensemble)',
        }


class OneClassSVMDetector:
    """One-Class SVM anomaly detector."""
    
    def __init__(
        self,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        nu: float = 0.1,
        max_train_samples: Optional[int] = 5000
    ):
        self.model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.max_train_samples = max_train_samples
        self.fit_time_ = 0.0
        self.n_train_actual_ = 0
    
    def fit(self, X: np.ndarray) -> 'OneClassSVMDetector':
        start = time.time()
        if self.max_train_samples and len(X) > self.max_train_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(X), self.max_train_samples, replace=False)
            X_train = X[indices]
            self.n_train_actual_ = self.max_train_samples
        else:
            X_train = X
            self.n_train_actual_ = len(X)
        
        self.model.fit(X_train)
        self.fit_time_ = time.time() - start
        return self
    
    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'model_type': 'OneClassSVM',
            'kernel': self.kernel,
            'gamma': str(self.gamma),
            'nu': self.nu,
            'max_train_samples': self.max_train_samples,
            'n_train_actual': self.n_train_actual_,
            'fit_time_seconds': self.fit_time_,
            'n_support_vectors': len(self.model.support_vectors_) if hasattr(self.model, 'support_vectors_') else 'N/A',
        }


def create_detector(model_type: str, **kwargs):
    detectors = {
        'statistical': StatisticalThresholdDetector,
        'isolation_forest': IsolationForestDetector,
        'ocsvm': OneClassSVMDetector,
    }
    if model_type not in detectors:
        raise ValueError(f"Unknown model type: {model_type}")
    return detectors[model_type](**kwargs)
