"""LSTM Autoencoder for temporal anomaly detection in multivariate time series.

Architecture:
  Input (batch, seq_len, n_features)
    -> LSTM Encoder
    -> Latent Vector (batch, latent_dim)
    -> RepeatVector (batch, seq_len, latent_dim)
    -> LSTM Decoder
    -> Linear Projection (batch, seq_len, n_features)

Anomaly Score: Reconstruction error (MSE / MAE) between input sequence and reconstructed sequence.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class LSTMEncoder(nn.Module):
    """LSTM-based sequence encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        enc_output_dim = hidden_dim * self.num_directions
        self.fc = nn.Linear(enc_output_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence (batch, seq_len, input_dim) to latent vector (batch, latent_dim)."""
        _, (h_n, _) = self.lstm(x)
        # Take the top layer hidden state
        if self.bidirectional:
            # Concatenate forward and backward top-layer hidden states
            h_top = torch.cat((h_n[-2], h_n[-1]), dim=-1)
        else:
            h_top = h_n[-1]
        latent = self.fc(h_top)
        return latent


class LSTMDecoder(nn.Module):
    """LSTM-based sequence decoder with RepeatVector mechanism."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 19,
        seq_len: int = 30,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector (batch, latent_dim) to sequence (batch, seq_len, output_dim)."""
        # Repeat vector along time dimension: (batch, seq_len, latent_dim)
        repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(repeated)
        reconstruction = self.fc(out)
        return reconstruction


class LSTMAutoencoder(nn.Module):
    """Full LSTM Autoencoder model."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 30,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMAutoencoderDetector:
    """LSTM Autoencoder anomaly detector with fit/score/get_params interface."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 30,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 100,
        patience: int = 15,
        clip_grad_norm: float = 1.0,
        loss_type: str = 'mse',
        temporal_aggregate: str = 'mean',
        device: str = 'cpu',
        seed: int = 42
    ):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.clip_grad_norm = clip_grad_norm
        self.loss_type = loss_type
        self.temporal_aggregate = temporal_aggregate
        self.device = device
        self.seed = seed

        self.model: Optional[LSTMAutoencoder] = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.fit_time_ = 0.0
        self.best_epoch_ = 0
        self.n_parameters_ = 0

    def fit(
        self,
        X_train_seq: np.ndarray,
        X_val_seq: Optional[np.ndarray] = None
    ) -> 'LSTMAutoencoderDetector':
        """Fit the LSTM-AE on normal sequence data."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        start = time.time()
        self.model = LSTMAutoencoder(
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)

        self.n_parameters_ = self.model.count_parameters()

        train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        val_tensor = None
        if X_val_seq is not None:
            val_tensor = torch.FloatTensor(X_val_seq).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        criterion = nn.MSELoss() if self.loss_type == 'mse' else nn.L1Loss()

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for (batch,) in train_loader:
                optimizer.zero_grad()
                reconstruction = self.model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()

                if self.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / n_batches
            self.train_losses.append(avg_train)

            if val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_recon = self.model(val_tensor)
                    val_loss = criterion(val_recon, val_tensor).item()
                self.val_losses.append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    self.best_epoch_ = epoch + 1
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"    Early stopping at epoch {epoch+1} (best: {self.best_epoch_}, val_loss: {best_val_loss:.6f})")
                    break
            else:
                if avg_train < best_val_loss:
                    best_val_loss = avg_train
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    self.best_epoch_ = epoch + 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_info = f", val={val_loss:.6f}" if val_tensor is not None else ""
                print(f"    Epoch {epoch+1:3d}: train={avg_train:.6f}{val_info}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.model.eval()
        self.fit_time_ = time.time() - start
        return self

    def score(self, X_seq: np.ndarray) -> np.ndarray:
        """Compute anomaly score per sequence."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        scores = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i:i + self.batch_size]
                recon = self.model(batch)

                if self.loss_type == 'mse':
                    diff = (batch - recon) ** 2
                else:
                    diff = torch.abs(batch - recon)

                # diff shape: (batch_size, seq_len, n_features)
                if self.temporal_aggregate == 'mean':
                    # Average over sequence length and features
                    err = torch.mean(diff, dim=(1, 2))
                elif self.temporal_aggregate == 'last':
                    # Error at the last timestep of the sequence
                    err = torch.mean(diff[:, -1, :], dim=1)
                elif self.temporal_aggregate == 'max':
                    err = torch.amax(torch.mean(diff, dim=2), dim=1)
                else:
                    err = torch.mean(diff, dim=(1, 2))

                scores.append(err.cpu().numpy())

        return np.concatenate(scores)

    def get_params(self) -> Dict[str, Any]:
        return {
            'model_type': 'LSTMAutoencoder',
            'input_dim': self.input_dim,
            'seq_len': self.seq_len,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'learning_rate': self.lr,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'patience': self.patience,
            'loss_type': self.loss_type,
            'temporal_aggregate': self.temporal_aggregate,
            'best_epoch': self.best_epoch_,
            'n_parameters': self.n_parameters_,
            'fit_time_seconds': self.fit_time_,
            'device': self.device,
        }

