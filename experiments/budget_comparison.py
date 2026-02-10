"""
Budget-Constrained Comparison
==============================

Fair comparison: given the SAME communication budget (same number of communications),
does uncertainty-aware choose BETTER moments to communicate than threshold?

Approach:
- Threshold: communicates when error > epsilon (reactive)
- Uncertainty-aware (budget-matched): communicates at the top-K highest uncertainty
  moments, where K = number of threshold communications

This directly tests: does uncertainty identify the "right" moments?
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import prepare_dataloaders
from src.model import TrajectoryLSTM, train_one_epoch, evaluate
from src.communication import compute_tracking_error, CommunicationResult
from src.evaluate import print_comparison_table, plot_pareto


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="D:/Datasets/tdrive")
    parser.add_argument("--results_dir", type=str, default="results/budget_comparison")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mc_samples", type=int, default=30)
    parser.add_argument("--epsilon", type=float, default=50.0)
    return parser.parse_args()


def denormalize(data_normalized, stats):
    return data_normalized * stats["std"] + stats["mean"]


def simulate_threshold_trajectory(true_pos, pred_pos, epsilon):
    """Threshold on a single trajectory. Returns per-step errors and comm indices."""
    T = len(true_pos)
    server_pos = np.zeros_like(true_pos)
    comm_indices = []

    server_pos[0] = true_pos[0]
    for t in range(1, T):
        server_pos[t] = pred_pos[t]
        error = np.sqrt(((true_pos[t] - pred_pos[t]) ** 2).sum())
        if error > epsilon:
            server_pos[t] = true_pos[t].copy()
            comm_indices.append(t)

    errors = compute_tracking_error(true_pos, server_pos)
    return errors, comm_indices


def simulate_topk_uncertainty(true_pos, pred_pos, uncertainties, k):
    """
    Budget-matched uncertainty: communicate at the top-K highest uncertainty moments.
    This gives uncertainty the SAME budget as threshold.
    """
    T = len(true_pos)
    if k <= 0 or k >= T:
        # If budget is 0 or >= T, just use predictions
        server_pos = pred_pos.copy()
        server_pos[0] = true_pos[0]
        errors = compute_tracking_error(true_pos, server_pos)
        return errors, []

    # Find top-K uncertainty indices (skip t=0)
    unc_with_idx = [(uncertainties[t], t) for t in range(1, T)]
    unc_with_idx.sort(reverse=True)
    comm_indices = sorted([idx for _, idx in unc_with_idx[:k]])

    server_pos = np.zeros_like(true_pos)
    server_pos[0] = true_pos[0]
    for t in range(1, T):
        if t in comm_indices:
            server_pos[t] = true_pos[t].copy()
        else:
            server_pos[t] = pred_pos[t]

    errors = compute_tracking_error(true_pos, server_pos)
    return errors, comm_indices


def simulate_random_comm(true_pos, pred_pos, k, rng):
    """Random baseline: communicate at K random moments."""
    T = len(true_pos)
    if k <= 0 or k >= T:
        server_pos = pred_pos.copy()
        server_pos[0] = true_pos[0]
        errors = compute_tracking_error(true_pos, server_pos)
        return errors, []

    comm_indices = sorted(rng.choice(range(1, T), size=min(k, T - 1), replace=False).tolist())

    server_pos = np.zeros_like(true_pos)
    server_pos[0] = true_pos[0]
    for t in range(1, T):
        if t in comm_indices:
            server_pos[t] = true_pos[t].copy()
        else:
            server_pos[t] = pred_pos[t]

    errors = compute_tracking_error(true_pos, server_pos)
    return errors, comm_indices


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.results_dir, exist_ok=True)

    # Load data
    print("\n=== Loading data ===")
    train_loader, val_loader, test_loader, stats = prepare_dataloaders(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
    )

    # Train model
    print("\n=== Training ===")
    model = TrajectoryLSTM(
        hidden_dim=args.hidden_dim,
        pred_len=args.pred_len,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    best_val, best_state = float("inf"), None

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

    model.load_state_dict(best_state)
    model = model.to(device)
    print(f"  Best val: {best_val:.6f}")

    # Collect test trajectories with predictions and uncertainty
    print("\n=== Collecting predictions ===")
    all_true, all_pred, all_unc = [], [], []

    for inputs, targets in test_loader:
        inputs_d = inputs.to(device)
        mean_pred, unc = model.predict_with_uncertainty(inputs_d, n_samples=args.mc_samples)
        mean_pred = mean_pred.cpu().numpy()
        unc = unc.cpu().numpy()
        targets_np = targets.numpy()

        mean_pred_m = denormalize(mean_pred, stats)
        targets_m = denormalize(targets_np, stats)
        avg_std = np.sqrt((stats["std"] ** 2).mean())
        unc_m = unc * avg_std

        for i in range(inputs.shape[0]):
            all_true.append(np.cumsum(targets_m[i], axis=0))
            all_pred.append(np.cumsum(mean_pred_m[i], axis=0))
            all_unc.append(unc_m[i])

    print(f"  {len(all_true)} test trajectories")

    # Run budget-matched comparison
    print("\n=== Budget-matched comparison ===")
    rng = np.random.default_rng(42)

    threshold_errors_all = []
    uncertainty_errors_all = []
    random_errors_all = []
    total_threshold_comms = 0
    total_unc_comms = 0
    total_random_comms = 0
    total_steps = 0

    for true_pos, pred_pos, unc in zip(all_true, all_pred, all_unc):
        # 1. Threshold baseline
        th_errors, th_comms = simulate_threshold_trajectory(
            true_pos, pred_pos, epsilon=args.epsilon
        )
        k = len(th_comms)  # budget = same as threshold

        # 2. Uncertainty top-K (same budget)
        unc_errors, unc_comms = simulate_topk_uncertainty(true_pos, pred_pos, unc, k)

        # 3. Random baseline (same budget)
        rand_errors, rand_comms = simulate_random_comm(true_pos, pred_pos, k, rng)

        threshold_errors_all.append(th_errors)
        uncertainty_errors_all.append(unc_errors)
        random_errors_all.append(rand_errors)

        total_threshold_comms += len(th_comms)
        total_unc_comms += len(unc_comms)
        total_random_comms += len(rand_comms)
        total_steps += len(true_pos)

    # Aggregate
    th_all = np.concatenate(threshold_errors_all)
    unc_all = np.concatenate(uncertainty_errors_all)
    rand_all = np.concatenate(random_errors_all)

    results = [
        CommunicationResult(
            strategy="threshold",
            n_communications=total_threshold_comms,
            n_total_steps=total_steps,
            communication_ratio=total_threshold_comms / total_steps,
            errors=th_all,
            mean_error=th_all.mean(),
            max_error=th_all.max(),
            p95_error=np.percentile(th_all, 95),
            communication_indices=[],
        ),
        CommunicationResult(
            strategy="unc_topK (same budget)",
            n_communications=total_unc_comms,
            n_total_steps=total_steps,
            communication_ratio=total_unc_comms / total_steps,
            errors=unc_all,
            mean_error=unc_all.mean(),
            max_error=unc_all.max(),
            p95_error=np.percentile(unc_all, 95),
            communication_indices=[],
        ),
        CommunicationResult(
            strategy="random (same budget)",
            n_communications=total_random_comms,
            n_total_steps=total_steps,
            communication_ratio=total_random_comms / total_steps,
            errors=rand_all,
            mean_error=rand_all.mean(),
            max_error=rand_all.max(),
            p95_error=np.percentile(rand_all, 95),
            communication_indices=[],
        ),
    ]

    print_comparison_table(results)
    plot_pareto(results, save_path=os.path.join(args.results_dir, "budget_pareto.png"))

    # Key result
    print("\n=== Key Result ===")
    print(f"  Same communication budget: {total_threshold_comms} comms / {total_steps} steps = {total_threshold_comms/total_steps:.3f}")
    print(f"  Threshold mean error:    {th_all.mean():.2f}m")
    print(f"  Uncertainty mean error:  {unc_all.mean():.2f}m")
    print(f"  Random mean error:       {rand_all.mean():.2f}m")

    if unc_all.mean() < th_all.mean():
        imp = (1 - unc_all.mean() / th_all.mean()) * 100
        print(f"  >>> Uncertainty selects BETTER moments: {imp:.1f}% lower error <<<")
    else:
        print(f"  >>> Threshold wins at equal budget — uncertainty not helpful <<<")

    # Save
    summary = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "budget": total_threshold_comms,
        "total_steps": total_steps,
        "results": {r.strategy: {"mean_error": float(r.mean_error), "p95_error": float(r.p95_error)} for r in results},
    }
    with open(os.path.join(args.results_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
