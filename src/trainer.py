"""
Model training, evaluation, and checkpoint management.

Extracted from model.py (training functions) and rolling_validation.py (training loop).
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from tqdm import tqdm

from src.model import (
    TrajectoryLSTM, HeteroscedasticLSTM, MDNTrajectoryLSTM,
    gaussian_nll_loss, mdn_nll_loss,
)
from src.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Training functions (moved from model.py)
# ---------------------------------------------------------------------------

def train_one_epoch(model, train_loader, optimizer, device,
                    epoch=0, total_epochs=0):
    """Train MSE model (TrajectoryLSTM) for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    criterion = nn.MSELoss()

    desc = f"Epoch {epoch+1}/{total_epochs} [train]" if total_epochs else "Training"
    pbar = tqdm(train_loader, desc=desc, leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.6f}")

    return total_loss / n_batches


def train_one_epoch_nll(model, train_loader, optimizer, device,
                        epoch=0, total_epochs=0):
    """Train HeteroscedasticLSTM with Gaussian NLL loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    desc = f"Epoch {epoch+1}/{total_epochs} [NLL]" if total_epochs else "Training"
    pbar = tqdm(train_loader, desc=desc, leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        mean, logvar = model(inputs)
        loss = gaussian_nll_loss(mean, logvar, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.6f}")

    return total_loss / n_batches


def train_one_epoch_mdn(model, train_loader, optimizer, device,
                        epoch=0, total_epochs=0):
    """Train MDNTrajectoryLSTM with mixture NLL loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    desc = f"Epoch {epoch+1}/{total_epochs} [MDN]" if total_epochs else "Training"
    pbar = tqdm(train_loader, desc=desc, leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        pi, mu, logvar = model(inputs)
        loss = mdn_nll_loss(pi, mu, logvar, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.6f}")

    return total_loss / n_batches


def evaluate_model(model, data_loader, device):
    """Evaluate model, return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            if isinstance(model, MDNTrajectoryLSTM):
                pi, mu, logvar = model(inputs)
                loss = mdn_nll_loss(pi, mu, logvar, targets)
            elif isinstance(model, HeteroscedasticLSTM):
                mean, logvar = model(inputs)
                loss = gaussian_nll_loss(mean, logvar, targets)
            else:
                predictions = model(inputs)
                loss = nn.MSELoss()(predictions, targets)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

_TRAIN_FNS = {
    "mcdropout": train_one_epoch,
    "heteroscedastic": train_one_epoch_nll,
    "mdn": train_one_epoch_mdn,
}


def build_model(config: ExperimentConfig, device: torch.device) -> nn.Module:
    """Construct model from config."""
    if config.model_type == "mcdropout":
        model = TrajectoryLSTM(
            hidden_dim=config.hidden_dim, pred_len=config.pred_len,
            dropout=config.dropout,
        )
    elif config.model_type == "heteroscedastic":
        model = HeteroscedasticLSTM(
            hidden_dim=config.hidden_dim, pred_len=config.pred_len,
            dropout=config.dropout,
        )
    elif config.model_type == "mdn":
        model = MDNTrajectoryLSTM(
            hidden_dim=config.hidden_dim, pred_len=config.pred_len,
            dropout=config.dropout, n_components=config.n_components,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    return model.to(device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    config: ExperimentConfig,
    train_loader,
    val_loader,
    device: torch.device,
    save_dir: str = None,
) -> Tuple[nn.Module, dict]:
    """
    Full training loop with early stopping.

    Returns:
        (model, stats_from_checkpoint) — model with best validation weights loaded
    """
    model = build_model(config, device)
    train_fn = _TRAIN_FNS[config.model_type]

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    best_val, best_state = float("inf"), None

    for epoch in range(config.epochs):
        train_loss = train_fn(
            model, train_loader, optimizer, device,
            epoch=epoch, total_epochs=config.epochs,
        )
        val_loss = evaluate_model(model, val_loader, device)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1}/{config.epochs}: "
              f"train={train_loss:.6f}, val={val_loss:.6f}, lr={lr:.1e}")

    model.load_state_dict(best_state)
    model = model.to(device)
    print(f"  Best val: {best_val:.6f}")

    # Save checkpoint
    if save_dir:
        ckpt_path = os.path.join(save_dir, "model.pt")
        torch.save({
            "model_state": model.state_dict(),
            "model_type": config.model_type,
            "config": {
                "hidden_dim": config.hidden_dim,
                "pred_len": config.pred_len,
                "dropout": config.dropout,
                "n_components": config.n_components,
            },
        }, ckpt_path)
        print(f"  Saved model to {ckpt_path}")

    return model


def load_checkpoint(
    config: ExperimentConfig,
    model_path: str,
    device: torch.device,
) -> Tuple[nn.Module, dict]:
    """
    Load model from checkpoint.

    Returns:
        (model, stats) where stats contains normalization parameters
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Override model_type from checkpoint if saved
    saved_type = checkpoint.get("model_type", config.model_type)
    config.model_type = saved_type

    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"  Loaded {saved_type} model from {model_path}")

    stats = checkpoint.get("stats", None)
    return model, stats
