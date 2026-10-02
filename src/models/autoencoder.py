"""Feedforward Autoencoder for anomaly detection.

Architecture: Input → Encoder → Bottleneck → Decoder → Reconstruction
Anomaly score: Per-sample reconstruction error (MSE)
Training: On normal/healthy data only

Configurable architecture via config dict.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple


class Autoencoder(nn.Module):
    """Configurable feedforward autoencoder."""
    
    def __init__(
        self,
        input_dim: int,
        encoder_dims: List[int] = [64, 32],
        latent_dim: int = 16,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """Initialize autoencoder.
        
        Args:
            input_dim: Number of input features.
            encoder_dims: Hidden layer sizes for encoder (decoder mirrors).
            latent_dim: Bottleneck dimension.
            dropout: Dropout rate.
            activation: Activation function ('relu', 'tanh', 'leaky_relu').
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        self.latent_dim = latent_dim
        self.dropout_rate = dropout
        
        act_fn = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
        }[activation]
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoder_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        decoder_dims = list(reversed(encoder_dims))
        prev_dim = latent_dim
        for dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode.
        
        Args:
            x: Input tensor of shape (batch, input_dim).
            
        Returns:
            Reconstructed tensor of shape (batch, input_dim).
        """
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AutoencoderDetector:
    """Autoencoder-based anomaly detector.
    
    Trains on normal data, uses reconstruction error as anomaly score.
    Follows the same interface as baseline detectors: fit/score/get_params.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoder_dims: List[int] = [64, 32],
        latent_dim: int = 16,
        dropout: float = 0.1,
        activation: str = 'relu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 10,
        device: str = 'cpu',
        seed: int = 42,
    ):
        """Initialize detector.
        
        Args:
            input_dim: Number of input features.
            encoder_dims: Encoder hidden layer sizes.
            latent_dim: Bottleneck dimension.
            dropout: Dropout rate.
            activation: Activation function.
            learning_rate: Optimizer learning rate.
            weight_decay: L2 regularization.
            batch_size: Training batch size.
            max_epochs: Maximum training epochs.
            patience: Early stopping patience.
            device: 'cpu' or 'cuda'.
            seed: Random seed.
        """
        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.activation = activation
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.seed = seed
        
        self.model: Optional[Autoencoder] = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.fit_time_ = 0.0
        self.best_epoch_ = 0
        self.n_parameters_ = 0
    
    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None
    ) -> 'AutoencoderDetector':
        """Train autoencoder on normal data.
        
        Args:
            X_train: Normal training data (n_samples, n_features).
            X_val: Validation data for early stopping (optional).
            
        Returns:
            Self.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        start = time.time()
        
        # Build model
        self.model = Autoencoder(
            input_dim=self.input_dim,
            encoder_dims=self.encoder_dims,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            activation=self.activation
        ).to(self.device)
        
        self.n_parameters_ = self.model.count_parameters()
        
        # Prepare data
        train_tensor = torch.FloatTensor(X_train).to(self.device)
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            drop_last=False
        )
        
        val_tensor = None
        if X_val is not None:
            val_tensor = torch.FloatTensor(X_val).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(self.max_epochs):
            # Train
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            for (batch,) in train_loader:
                optimizer.zero_grad()
                reconstruction = self.model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            self.train_losses.append(avg_train_loss)
            
            # Validate
            if val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_recon = self.model(val_tensor)
                    val_loss = criterion(val_recon, val_tensor).item()
                self.val_losses.append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    self.best_epoch_ = epoch + 1
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    print(f"    Early stopping at epoch {epoch+1} "
                          f"(best: {self.best_epoch_}, val_loss: {best_val_loss:.6f})")
                    break
            else:
                # No validation: just track best train loss
                if avg_train_loss < best_val_loss:
                    best_val_loss = avg_train_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    self.best_epoch_ = epoch + 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_str = f", val={val_loss:.6f}" if val_tensor is not None else ""
                print(f"    Epoch {epoch+1:3d}: train={avg_train_loss:.6f}{val_str}")
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        
        self.model.eval()
        self.fit_time_ = time.time() - start
        
        return self
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample reconstruction error (MSE).
        
        Args:
            X: Data array (n_samples, n_features).
            
        Returns:
            Anomaly scores (higher = more anomalous).
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            # Process in batches to avoid memory issues
            scores = []
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i:i + self.batch_size]
                reconstruction = self.model(batch)
                # Per-sample MSE
                mse = torch.mean((batch - reconstruction) ** 2, dim=1)
                scores.append(mse.cpu().numpy())
        
        return np.concatenate(scores)
    
    def score_per_feature(self, X: np.ndarray) -> np.ndarray:
        """Compute per-feature reconstruction error.
        
        Useful for identifying which features contribute most to anomaly.
        
        Args:
            X: Data array (n_samples, n_features).
            
        Returns:
            Per-feature errors (n_samples, n_features).
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            errors = []
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i:i + self.batch_size]
                reconstruction = self.model(batch)
                err = (batch - reconstruction) ** 2
                errors.append(err.cpu().numpy())
        
        return np.concatenate(errors)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters for logging."""
        return {
            'model_type': 'Autoencoder',
            'input_dim': self.input_dim,
            'encoder_dims': self.encoder_dims,
            'latent_dim': self.latent_dim,
            'dropout': self.dropout,
            'activation': self.activation,
            'learning_rate': self.lr,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'patience': self.patience,
            'best_epoch': self.best_epoch_,
            'n_parameters': self.n_parameters_,
            'fit_time_seconds': self.fit_time_,
            'device': self.device,
        }
