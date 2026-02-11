"""
Trajectory prediction models with uncertainty estimation.

4 model variants = 2 backbones × 2 output heads:

| Backbone \ Head |  Point (MSE)  |     MDN      |
|-----------------|---------------|--------------|
| LSTM            | LSTM          | LSTM_MDN     |
| Transformer     | Transformer   | Transformer_MDN |

Point models:
    forward(x) -> (batch, pred_len, 2)
    predict_mean(x) -> (batch, pred_len, 2)

MDN models:
    forward(x) -> (pi, mu, logvar)
    predict_mean(x) -> (batch, pred_len, 2)  — mixture mean
    predict_with_uncertainty(x) -> (mean, std_per_dim)  — Law of Total Variance
"""

import math
import torch
import torch.nn as nn
from typing import Tuple


# =====================================================================
# Helper: Positional Encoding (shared by Transformer variants)
# =====================================================================

class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =====================================================================
# Helper: MDN predict_with_uncertainty (shared by MDN variants)
# =====================================================================

def _mdn_predict_with_uncertainty(
    pi: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Law of Total Variance for MDN uncertainty.

    Returns:
        mean: (B, P, D) — mixture mean
        std:  (B, P, D) — per-dimension total std
    """
    var = torch.exp(logvar)           # (B, P, K, D)
    pi_e = pi.unsqueeze(-1)           # (B, P, K, 1)

    mixture_mean = (pi_e * mu).sum(dim=2)                    # (B, P, D)
    aleatoric = (pi_e * var).sum(dim=2)                      # E[sigma^2]
    epistemic = (pi_e * (mu - mixture_mean.unsqueeze(2)) ** 2).sum(dim=2)  # Var[mu]

    return mixture_mean, torch.sqrt(aleatoric + epistemic)


def _mdn_predict_mean(
    pi: torch.Tensor, mu: torch.Tensor,
) -> torch.Tensor:
    """Mixture mean from MDN outputs."""
    pi_e = pi.unsqueeze(-1)
    return (pi_e * mu).sum(dim=2)


# =====================================================================
# Model 1: LSTM (Point prediction, MSE loss)
# =====================================================================

class LSTM(nn.Module):
    """
    LSTM backbone + point output head.

    Input:  (batch, seq_len, 2) displacement sequence
    Output: (batch, pred_len, 2) predicted displacements
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_len: int = 1,
        dropout: float = 0.2,
        **kwargs,  # ignore MDN/Transformer params
    ):
        super().__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, pred_len * input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out.view(-1, self.pred_len, self.input_dim)

    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)


# =====================================================================
# Model 2: LSTM_MDN (LSTM + Mixture Density Network)
# =====================================================================

class LSTM_MDN(nn.Module):
    """
    LSTM backbone + MDN output head.

    Multi-modal prediction with uncertainty via Law of Total Variance.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_len: int = 1,
        dropout: float = 0.2,
        n_components: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.n_components = n_components

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        K = n_components
        self.fc_pi = nn.Linear(hidden_dim, pred_len * K)
        self.fc_mu = nn.Linear(hidden_dim, pred_len * K * input_dim)
        self.fc_logvar = nn.Linear(hidden_dim, pred_len * K * input_dim)

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        K, P, D = self.n_components, self.pred_len, self.input_dim

        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        pi = torch.softmax(self.fc_pi(last_hidden).view(B, P, K), dim=-1)
        mu = self.fc_mu(last_hidden).view(B, P, K, D)
        logvar = torch.clamp(self.fc_logvar(last_hidden).view(B, P, K, D), -10, 10)

        return pi, mu, logvar

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            pi, mu, logvar = self.forward(x)
            return _mdn_predict_with_uncertainty(pi, mu, logvar)

    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            pi, mu, _ = self.forward(x)
            return _mdn_predict_mean(pi, mu)


# =====================================================================
# Model 3: Transformer (Point prediction, MSE loss)
# =====================================================================

class Transformer(nn.Module):
    """
    Transformer encoder backbone + point output head.

    Input:  (batch, seq_len, 2) displacement sequence
    Output: (batch, pred_len, 2) predicted displacements
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        pred_len: int = 1,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = _PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(hidden_dim, pred_len * input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos_enc(self.input_proj(x))
        h = self.transformer(h)
        last_hidden = h[:, -1, :]
        out = self.fc(last_hidden)
        return out.view(-1, self.pred_len, self.input_dim)

    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)


# =====================================================================
# Model 4: Transformer_MDN (Transformer + Mixture Density Network)
# =====================================================================

class Transformer_MDN(nn.Module):
    """
    Transformer encoder backbone + MDN output head.

    Multi-modal prediction with uncertainty via Law of Total Variance.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        pred_len: int = 1,
        dropout: float = 0.1,
        n_components: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.n_components = n_components

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = _PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        K = n_components
        self.fc_pi = nn.Linear(hidden_dim, pred_len * K)
        self.fc_mu = nn.Linear(hidden_dim, pred_len * K * input_dim)
        self.fc_logvar = nn.Linear(hidden_dim, pred_len * K * input_dim)

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        K, P, D = self.n_components, self.pred_len, self.input_dim

        h = self.pos_enc(self.input_proj(x))
        h = self.transformer(h)
        last_hidden = h[:, -1, :]

        pi = torch.softmax(self.fc_pi(last_hidden).view(B, P, K), dim=-1)
        mu = self.fc_mu(last_hidden).view(B, P, K, D)
        logvar = torch.clamp(self.fc_logvar(last_hidden).view(B, P, K, D), -10, 10)

        return pi, mu, logvar

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            pi, mu, logvar = self.forward(x)
            return _mdn_predict_with_uncertainty(pi, mu, logvar)

    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            pi, mu, _ = self.forward(x)
            return _mdn_predict_mean(pi, mu)


# =====================================================================
# Model registry
# =====================================================================

MODEL_CLASSES = {
    "lstm": LSTM,
    "lstm_mdn": LSTM_MDN,
    "transformer": Transformer,
    "transformer_mdn": Transformer_MDN,
}

# Which model types use MDN output (pi, mu, logvar)
MDN_MODELS = {"lstm_mdn", "transformer_mdn"}

# Which model types support predict_with_uncertainty
UNCERTAINTY_MODELS = MDN_MODELS  # only MDN models for now


# =====================================================================
# Loss functions
# =====================================================================

def mdn_nll_loss(pi, mu, logvar, target):
    """
    Negative log-likelihood of mixture of Gaussians.

    Args:
        pi: (B, P, K) mixing coefficients
        mu: (B, P, K, D) component means
        logvar: (B, P, K, D) component log-variances
        target: (B, P, D) ground truth
    """
    var = torch.exp(logvar)
    target_e = target.unsqueeze(2)  # (B, P, 1, D)

    log_comp = -0.5 * (logvar + (target_e - mu) ** 2 / var)  # (B, P, K, D)
    log_comp = log_comp.sum(dim=-1)  # (B, P, K)

    log_mix = torch.log(pi + 1e-8) + log_comp  # (B, P, K)
    log_likelihood = torch.logsumexp(log_mix, dim=-1)  # (B, P)

    return -log_likelihood.mean()
