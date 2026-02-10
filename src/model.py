"""
Trajectory prediction models with uncertainty estimation.

Three model variants:
1. TrajectoryLSTM: MC Dropout baseline (multiple forward passes)
2. HeteroscedasticLSTM: Direct variance prediction (single pass)
3. MDNTrajectoryLSTM: Mixture Density Network (single pass, multi-modal)

All models share the same LSTM backbone and predict_with_uncertainty API:
    mean: (batch, pred_len, 2)
    std_per_dim: (batch, pred_len, 2)  — per-dimension std in normalized space
"""

import torch
import torch.nn as nn
from typing import Tuple


# ---------------------------------------------------------------------------
# Model 1: MC Dropout baseline
# ---------------------------------------------------------------------------

class TrajectoryLSTM(nn.Module):
    """
    LSTM-based trajectory prediction with MC Dropout for uncertainty.

    Input: (batch, seq_len, 2) - historical displacement sequence
    Output: (batch, pred_len, 2) - predicted future displacements
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_len: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.input_dim = input_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Dropout layer for MC Dropout (applied during both training and inference)
        self.mc_dropout = nn.Dropout(p=dropout)

        # Output projection: hidden -> pred_len * 2
        self.fc = nn.Linear(hidden_dim, pred_len * input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.mc_dropout(last_hidden)
        out = self.fc(last_hidden)
        out = out.view(-1, self.pred_len, 2)
        return out

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC Dropout: multiple forward passes with dropout enabled.

        Returns:
            mean_pred: (batch, pred_len, 2)
            std_per_dim: (batch, pred_len, 2) per-dimension std
        """
        self.train()  # Enable dropout

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        std_per_dim = predictions.std(dim=0)

        return mean_pred, std_per_dim

    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass, no dropout."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


# ---------------------------------------------------------------------------
# Model 2: Heteroscedastic (direct variance prediction)
# ---------------------------------------------------------------------------

class HeteroscedasticLSTM(nn.Module):
    """
    LSTM that predicts both mean and log-variance.
    Single forward pass for uncertainty — no MC sampling needed.

    Captures aleatoric (data-inherent) uncertainty.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_len: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.input_dim = input_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc_mean = nn.Linear(hidden_dim, pred_len * input_dim)
        self.fc_logvar = nn.Linear(hidden_dim, pred_len * input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: (batch, pred_len, 2)
            logvar: (batch, pred_len, 2) — clamped for stability
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        mean = self.fc_mean(last_hidden).view(-1, self.pred_len, self.input_dim)
        logvar = self.fc_logvar(last_hidden).view(-1, self.pred_len, self.input_dim)
        logvar = torch.clamp(logvar, min=-10, max=10)

        return mean, logvar

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single forward pass uncertainty.

        Returns:
            mean: (batch, pred_len, 2)
            std_per_dim: (batch, pred_len, 2)
        """
        self.eval()
        with torch.no_grad():
            mean, logvar = self.forward(x)
            std = torch.exp(0.5 * logvar)
        return mean, std

    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            mean, _ = self.forward(x)
        return mean


# ---------------------------------------------------------------------------
# Model 3: Mixture Density Network
# ---------------------------------------------------------------------------

class MDNTrajectoryLSTM(nn.Module):
    """
    LSTM + Mixture of Gaussians output for multi-modal trajectory prediction.

    At intersections, different components capture different possible movements
    (straight, left turn, right turn). Single forward pass.

    Uncertainty via Law of Total Variance:
        Var_total = E[sigma^2] + Var[mu]
                  = aleatoric    + epistemic (inter-modal)
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_len: int = 5,
        dropout: float = 0.2,
        n_components: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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
        """
        Returns:
            pi: (batch, pred_len, K) mixing coefficients (softmax)
            mu: (batch, pred_len, K, 2) component means
            logvar: (batch, pred_len, K, 2) component log-variances
        """
        B = x.shape[0]
        K = self.n_components
        P = self.pred_len
        D = self.input_dim

        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        pi = self.fc_pi(last_hidden).view(B, P, K)
        pi = torch.softmax(pi, dim=-1)

        mu = self.fc_mu(last_hidden).view(B, P, K, D)

        logvar = self.fc_logvar(last_hidden).view(B, P, K, D)
        logvar = torch.clamp(logvar, min=-10, max=10)

        return pi, mu, logvar

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single forward pass. Law of Total Variance for uncertainty.

        Returns:
            mean: (batch, pred_len, 2) — mixture mean
            std_per_dim: (batch, pred_len, 2) — per-dimension total std
        """
        self.eval()
        with torch.no_grad():
            pi, mu, logvar = self.forward(x)
            var = torch.exp(logvar)  # (B, P, K, D)

            pi_e = pi.unsqueeze(-1)  # (B, P, K, 1)

            # Mixture mean
            mixture_mean = (pi_e * mu).sum(dim=2)  # (B, P, D)

            # Law of total variance
            aleatoric = (pi_e * var).sum(dim=2)  # E[sigma^2]
            diff = mu - mixture_mean.unsqueeze(2)
            epistemic = (pi_e * diff ** 2).sum(dim=2)  # Var[mu]

            total_std = torch.sqrt(aleatoric + epistemic)

        return mixture_mean, total_std

    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            pi, mu, _ = self.forward(x)
            pi_e = pi.unsqueeze(-1)
            mixture_mean = (pi_e * mu).sum(dim=2)
        return mixture_mean


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def gaussian_nll_loss(mean, logvar, target):
    """
    Gaussian NLL for heteroscedastic model.
    loss = 0.5 * (logvar + (target - mean)^2 / exp(logvar))
    """
    var = torch.exp(logvar)
    return 0.5 * (logvar + (target - mean) ** 2 / var).mean()


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

    # Log probability per component (diagonal covariance)
    log_comp = -0.5 * (logvar + (target_e - mu) ** 2 / var)  # (B, P, K, D)
    log_comp = log_comp.sum(dim=-1)  # sum over dims -> (B, P, K)

    # Add log mixing coefficients
    log_mix = torch.log(pi + 1e-8) + log_comp  # (B, P, K)

    # Log-sum-exp over components
    log_likelihood = torch.logsumexp(log_mix, dim=-1)  # (B, P)

    return -log_likelihood.mean()
