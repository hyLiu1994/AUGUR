"""
AUGUR main experiment runner.

Usage:
    python experiments/run_experiment.py --config configs/geolife_mdn.yaml
    python experiments/run_experiment.py --config configs/geolife_mdn.yaml --data_fraction 0.1
    python experiments/run_experiment.py --config configs/geolife_mdn.yaml --load_model results/.../model.pt
"""

import os
import sys
import csv
import json
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, save_config, create_run_dir
from src.data_loader import load_and_segment, create_sequences, TrajectoryDataset
from src.trainer import train_model, load_checkpoint
from src.simulator import (
    prepare_long_trajectories, split_trajectories,
    profile_uncertainty, run_all_strategies,
)
from src.strategies import CommunicationResult
from src.evaluate import print_comparison_table, plot_pareto, plot_uncertainty_vs_error


def _needs_model(config):
    """Check if any selected strategy requires a trained model."""
    NO_MODEL = {"dead_reckoning", "kalman_dps"}
    selected = config.strategies_list
    if selected is None:  # "all"
        return True
    return any(s not in NO_MODEL for s in selected)


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {config.model_type}")

    # Create run directory
    run_dir = create_run_dir(config)
    save_config(config, os.path.join(run_dir, "config.yaml"))
    print(f"Run directory: {run_dir}")

    # === Step 1: Prepare data ===
    print("\n=== Step 1: Prepare trajectories ===")
    trajectories = prepare_long_trajectories(config)
    train_trajs, test_trajs = split_trajectories(trajectories, config)

    needs_model = _needs_model(config)
    model, stats, median_unc = None, None, 0.0

    # === Step 2: Load or train model (skip if only model-free strategies) ===
    if not needs_model:
        print("\n=== Step 2: Skipped (no model needed for selected strategies) ===")
    elif config.load_model:
        print(f"\n=== Step 2: Loading model from {config.load_model} ===")
        model, stats = load_checkpoint(config, config.load_model, device)
    else:
        print(f"\n=== Step 2: Train {config.model_type} model ===")
        # Prepare training data
        all_segments = load_and_segment(config.data_dir, min_segment_len=20)
        if config.data_fraction < 1.0:
            rng = np.random.default_rng(config.seed)
            n_seg_use = max(10, int(len(all_segments) * config.data_fraction))
            seg_indices = rng.permutation(len(all_segments))[:n_seg_use]
            all_segments = [all_segments[i] for i in seg_indices]
            print(f"  Using {config.data_fraction:.0%} of segments: {len(all_segments)}")

        inputs, targets, stats = create_sequences(
            all_segments, seq_len=config.seq_len, pred_len=config.pred_len
        )

        n_total = len(inputs)
        n_train_seq = int(n_total * 0.85)
        rng = np.random.default_rng(config.seed)
        perm = rng.permutation(n_total)
        inputs, targets = inputs[perm], targets[perm]

        train_ds = TrajectoryDataset(inputs[:n_train_seq], targets[:n_train_seq])
        val_ds = TrajectoryDataset(inputs[n_train_seq:], targets[n_train_seq:])

        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)} sequences")

        model = train_model(config, train_loader, val_loader, device, save_dir=run_dir)

        # Save stats alongside model
        ckpt_path = os.path.join(run_dir, "model.pt")
        ckpt = torch.load(ckpt_path, weights_only=False)
        ckpt["stats"] = stats
        torch.save(ckpt, ckpt_path)

    # === Step 3: Profile uncertainty ===
    if needs_model:
        median_unc = profile_uncertainty(model, train_trajs, stats, config, device)
    else:
        print("\n=== Step 3: Skipped (no model) ===")

    # === Step 4: Simulate ===
    print(f"\n=== Step 4: Rolling simulation on {len(test_trajs)} test trajectories ===")
    all_results = run_all_strategies(model, test_trajs, stats, config, device, median_unc)

    # === Step 5: Report ===
    print("\n=== Results ===")
    result_objects = []
    for name, r in all_results.items():
        ro = CommunicationResult(
            strategy=name,
            n_communications=r["n_comms"],
            n_total_steps=r["n_steps"],
            communication_ratio=r["comm_ratio"],
            errors=np.array([r["mean_error"]]),
            mean_error=r["mean_error"],
            max_error=r["max_error"],
            p95_error=r["p95_error"],
            communication_indices=[],
        )
        result_objects.append(ro)

    print_comparison_table(result_objects)
    plot_pareto(result_objects, save_path=os.path.join(run_dir, "pareto.png"))

    # Calibration plot
    unc_keys = sorted([k for k in all_results
                       if k.startswith(("proactive_", "pronorm_", "dynamic_"))])
    if unc_keys:
        cal_key = unc_keys[0]
        cal_unc = all_results[cal_key]["uncertainties"]
        cal_err = all_results[cal_key]["pred_errors"]
        mask = cal_unc > 0
        if mask.sum() > 0:
            plot_uncertainty_vs_error(
                cal_unc[mask], cal_err[mask],
                save_path=os.path.join(run_dir, "calibration.png"),
            )

    # Save results JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "run_dir": run_dir,
        "config": {k: v for k, v in vars(config).items()},
        "stats": {"mean": stats["mean"].tolist(), "std": stats["std"].tolist()} if stats else None,
        "results": {
            name: {k: v for k, v in r.items() if k not in ("uncertainties", "pred_errors")}
            for name, r in all_results.items()
        },
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Append to CSV log
    csv_path = config.csv_log
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    run_id = os.path.basename(run_dir)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "run_id", "timestamp", "model_type", "dataset", "data_fraction", "epochs",
                "strategy", "n_comms", "n_steps", "comm_ratio",
                "mean_error_m", "max_error_m", "p95_error_m", "notes",
            ])
        ts = datetime.now().isoformat(timespec="seconds")
        dataset_name = os.path.basename(config.data_dir.rstrip("/\\")).lower()
        for name, r in all_results.items():
            writer.writerow([
                run_id, ts, config.model_type, dataset_name,
                config.data_fraction, config.epochs,
                name, r["n_comms"], r["n_steps"], f"{r['comm_ratio']:.4f}",
                f"{r['mean_error']:.2f}", f"{r['max_error']:.2f}", f"{r['p95_error']:.2f}",
                config.note,
            ])
    print(f"Appended {len(all_results)} rows to {csv_path}")

    # Key comparison
    print("\n=== Key Comparison ===")
    th_results = {k: v for k, v in all_results.items() if k.startswith("threshold_")}
    unc_results = {k: v for k, v in all_results.items()
                   if k.startswith(("proactive_", "pronorm_", "dynamic_"))}

    if th_results:
        for name, r in sorted(th_results.items(), key=lambda x: x[1]["comm_ratio"], reverse=True):
            print(f"  {name}: comm={r['comm_ratio']:.3f}, mean={r['mean_error']:.2f}m")

    if unc_results and th_results:
        print()
        for name, r in sorted(unc_results.items(), key=lambda x: x[1]["comm_ratio"], reverse=True):
            best_match = min(th_results.items(),
                             key=lambda x: abs(x[1]["comm_ratio"] - r["comm_ratio"]))
            th_name, th_r = best_match
            delta = r["mean_error"] - th_r["mean_error"]
            sign = "+" if delta > 0 else ""
            print(f"  {name}: comm={r['comm_ratio']:.3f}, mean={r['mean_error']:.2f}m "
                  f"(vs {th_name}: {sign}{delta:.2f}m)")

    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
