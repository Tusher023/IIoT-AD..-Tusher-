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
            'n_parameters': self.n_features_ * 2,  # mean + std per feature
        }


class IsolationForestDetector:
    """Isolation Forest anomaly detector.
    
    Ensemble of isolation trees that isolate anomalies through
    random partitioning. Anomalies are isolated in fewer steps.
    
    Reference: Liu, Ting, Zhou (2008). Isolation Forest. ICDM.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        contamination: str = 'auto',
        max_samples: str = 'auto',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """Initialize Isolation Forest.
        
        Args:
            n_estimators: Number of isolation trees.
            contamination: Expected fraction of anomalies ('auto' or float).
            max_samples: Samples per tree ('auto' or int).
            random_state: Random seed.
            n_jobs: Parallel jobs (-1 for all cores).
        """
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
        """Fit on normal/healthy data.
        
        Args:
            X: Normal data array of shape (n_samples, n_features).
            
        Returns:
            Self.
        """
        start = time.time()
        self.model.fit(X)
        self.fit_time_ = time.time() - start
        return self
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores.
        
        Converts sklearn's score (where lower = more anomalous) to
        our convention (higher = more anomalous) by negation.
        
        Args:
            X: Data array of shape (n_samples, n_features).
            
        Returns:
            Anomaly scores (higher = more anomalous).
        """
        # sklearn returns negative scores (lower = more anomalous)
        # We negate so higher = more anomalous
        return -self.model.score_samples(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters for logging."""
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
    """One-Class SVM anomaly detector.
    
    Learns a decision boundary around normal data in kernel space.
    Computationally expensive for large datasets.
    
    Note: May be infeasible for large C-MAPSS subsets. Use subsample
    parameter to limit training data if needed.
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        nu: float = 0.1,
        max_train_samples: Optional[int] = 5000
    ):
        """Initialize One-Class SVM.
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly').
            gamma: Kernel coefficient ('scale', 'auto', or float).
            nu: Upper bound on fraction of training errors.
            max_train_samples: Maximum samples for training (None=no limit).
                OC-SVM scales O(n^2)-O(n^3), so we cap training size.
        """
        self.model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.max_train_samples = max_train_samples
        self.fit_time_ = 0.0
        self.n_train_actual_ = 0
    
    def fit(self, X: np.ndarray) -> 'OneClassSVMDetector':
        """Fit on normal/healthy data.
        
        Subsamples if data exceeds max_train_samples for computational
        feasibility (OC-SVM has O(n^2) to O(n^3) complexity).
        
        Args:
            X: Normal data array of shape (n_samples, n_features).
            
        Returns:
            Self.
        """
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
        """Compute anomaly scores.
        
        Converts sklearn's decision_function (positive = inlier) to
        our convention (higher = more anomalous) by negation.
        
        Args:
            X: Data array of shape (n_samples, n_features).
            
        Returns:
            Anomaly scores (higher = more anomalous).
        """
        # sklearn returns positive for inliers, negative for outliers
        # We negate so higher = more anomalous
        return -self.model.decision_function(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters for logging."""
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
    """Factory function to create a detector by name.
    
    Args:
        model_type: 'statistical', 'isolation_forest', or 'ocsvm'.
        **kwargs: Model-specific parameters.
        
    Returns:
        Detector instance.
    """
    detectors = {
        'statistical': StatisticalThresholdDetector,
        'isolation_forest': IsolationForestDetector,
        'ocsvm': OneClassSVMDetector,
    }
    
    if model_type not in detectors:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(detectors.keys())}")
    
    return detectors[model_type](**kwargs)
