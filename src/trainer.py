"""
Model training, evaluation, and checkpoint management.

Supports Scheduled Sampling: during training, some input steps are replaced
with the model's own predictions to bridge the train-test gap. In dual
prediction, the server's input buffer contains predicted (noisy) values
between communications. Scheduled Sampling teaches the model to handle this.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from tqdm import tqdm

from src.model import MODEL_CLASSES, MDN_MODELS, mdn_nll_loss
from src.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Scheduled Sampling helper
# ---------------------------------------------------------------------------

def _get_ss_ratio(epoch: int, total_epochs: int, ss_max: float) -> float:
    """Linearly ramp scheduled sampling ratio from 0 to ss_max."""
    if total_epochs <= 1:
        return 0.0
    return ss_max * epoch / (total_epochs - 1)


def _apply_scheduled_sampling(model, inputs, ss_ratio, is_mdn):
    """
    Replace the last step of the input with the model's own prediction.

    In dual prediction, the most recent buffer entry is most likely to be a
    model prediction (no communication happened). We simulate this by:
    1. Forward pass on inputs[:, :-1, :] (first seq_len-1 steps)
    2. Get the model's predicted displacement
    3. With probability ss_ratio, replace inputs[:, -1, :] with prediction
    """
    if ss_ratio <= 0.0:
        return inputs

    B = inputs.shape[0]
    mask = torch.rand(B, device=inputs.device) < ss_ratio
    if not mask.any():
        return inputs

    prefix = inputs[:, :-1, :]

    with torch.no_grad():
        if is_mdn:
            pi, mu, _ = model(prefix)
            pi_e = pi.unsqueeze(-1)
            pred = (pi_e * mu).sum(dim=2)  # mixture mean
        else:
            pred = model(prefix)

    pred_step = pred[:, 0, :]

    inputs = inputs.clone()
    inputs[mask, -1, :] = pred_step[mask]

    return inputs


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_one_epoch(model, train_loader, optimizer, device,
                    epoch=0, total_epochs=0, ss_ratio=0.0):
    """Train point model (LSTM / Transformer) with MSE loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    criterion = nn.MSELoss()

    desc = f"Epoch {epoch+1}/{total_epochs} [MSE]" if total_epochs else "Training"
    pbar = tqdm(train_loader, desc=desc, leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if ss_ratio > 0:
            inputs = _apply_scheduled_sampling(model, inputs, ss_ratio, is_mdn=False)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.6f}")

    return total_loss / n_batches


def train_one_epoch_mdn(model, train_loader, optimizer, device,
                        epoch=0, total_epochs=0, ss_ratio=0.0):
    """Train MDN model (LSTM_MDN / Transformer_MDN) with mixture NLL loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    desc = f"Epoch {epoch+1}/{total_epochs} [MDN]" if total_epochs else "Training"
    pbar = tqdm(train_loader, desc=desc, leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if ss_ratio > 0:
            inputs = _apply_scheduled_sampling(model, inputs, ss_ratio, is_mdn=True)

        optimizer.zero_grad()
        pi, mu, logvar = model(inputs)
        loss = mdn_nll_loss(pi, mu, logvar, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.6f}")

    return total_loss / n_batches


def evaluate_model(model, data_loader, device, model_type):
    """Evaluate model, return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            if model_type in MDN_MODELS:
                pi, mu, logvar = model(inputs)
                loss = mdn_nll_loss(pi, mu, logvar, targets)
            else:
                predictions = model(inputs)
                loss = nn.MSELoss()(predictions, targets)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(config: ExperimentConfig, device: torch.device) -> nn.Module:
    """Construct model from config using MODEL_CLASSES registry."""
    model_type = config.model.type
    if model_type not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Valid types: {list(MODEL_CLASSES.keys())}"
        )

    cls = MODEL_CLASSES[model_type]
    model = cls(
        hidden_dim=config.model.hidden_dim,
        pred_len=config.model.pred_len,
        dropout=config.model.dropout,
        n_components=config.model.n_components,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
    )
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
) -> nn.Module:
    """
    Full training loop with early stopping.

    Returns:
        model with best validation weights loaded
    """
    model = build_model(config, device)
    model_type = config.model.type
    train_fn = train_one_epoch_mdn if model_type in MDN_MODELS else train_one_epoch

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    best_val, best_state = float("inf"), None

    for epoch in range(config.training.epochs):
        ss_ratio = _get_ss_ratio(epoch, config.training.epochs, config.training.ss_max)

        train_loss = train_fn(
            model, train_loader, optimizer, device,
            epoch=epoch, total_epochs=config.training.epochs,
            ss_ratio=ss_ratio,
        )
        val_loss = evaluate_model(model, val_loader, device, model_type)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        lr = optimizer.param_groups[0]["lr"]
        ss_str = f", ss={ss_ratio:.2f}" if config.training.ss_max > 0 else ""
        print(f"  Epoch {epoch+1}/{config.training.epochs}: "
              f"train={train_loss:.6f}, val={val_loss:.6f}, lr={lr:.1e}{ss_str}")

    model.load_state_dict(best_state)
    model = model.to(device)
    print(f"  Best val: {best_val:.6f}")

    # Save checkpoint
    if save_dir:
        ckpt_path = os.path.join(save_dir, "model.pt")
        torch.save({
            "model_state": model.state_dict(),
            "model_type": config.model.type,
            "config": {
                "hidden_dim": config.model.hidden_dim,
                "pred_len": config.model.pred_len,
                "dropout": config.model.dropout,
                "n_components": config.model.n_components,
                "n_heads": config.model.n_heads,
                "n_layers": config.model.n_layers,
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

    saved_type = checkpoint.get("model_type", config.model.type)
    config.model.type = saved_type

    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"  Loaded {saved_type} model from {model_path}")

    stats = checkpoint.get("stats", None)
    return model, stats
