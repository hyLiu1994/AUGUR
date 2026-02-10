"""
Evaluation metrics and visualization for communication experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from src.strategies import CommunicationResult


def plot_error_comparison(
    results: List[CommunicationResult],
    save_path: Optional[str] = None,
):
    """
    Plot tracking error over time for different strategies.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: tracking error
    ax1 = axes[0]
    for r in results:
        ax1.plot(r.errors, label=r.strategy, alpha=0.8)
    ax1.set_ylabel("Tracking Error (m)")
    ax1.set_title("Tracking Error Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom: communication events
    ax2 = axes[1]
    for i, r in enumerate(results):
        comm_times = r.communication_indices
        ax2.scatter(
            comm_times,
            [i] * len(comm_times),
            marker="|",
            s=50,
            label=r.strategy,
        )
    ax2.set_ylabel("Strategy")
    ax2.set_xlabel("Time Step")
    ax2.set_title("Communication Events")
    ax2.set_yticks(range(len(results)))
    ax2.set_yticklabels([r.strategy for r in results])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.close()


def plot_pareto(
    results: List[CommunicationResult],
    save_path: Optional[str] = None,
):
    """
    Pareto plot: communication ratio vs mean tracking error.
    This is the key plot showing the trade-off.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        ax.scatter(
            r.communication_ratio,
            r.mean_error,
            s=100,
            zorder=5,
        )
        ax.annotate(
            r.strategy,
            (r.communication_ratio, r.mean_error),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
        )

    ax.set_xlabel("Communication Ratio (lower = less bandwidth)")
    ax.set_ylabel("Mean Tracking Error in meters (lower = more accurate)")
    ax.set_title("Communication-Error Trade-off (Pareto)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved Pareto plot to {save_path}")
    plt.close()


def plot_uncertainty_vs_error(
    uncertainties: np.ndarray,
    actual_errors: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Scatter plot: predicted uncertainty vs actual error.
    Shows whether uncertainty is a good predictor of error.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(uncertainties, actual_errors, alpha=0.1, s=5)

    # Add diagonal reference
    max_val = max(uncertainties.max(), actual_errors.max())
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="y=x")

    ax.set_xlabel("Predicted Uncertainty (m)")
    ax.set_ylabel("Actual Error (m)")
    ax.set_title("Uncertainty Calibration: Predicted vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved calibration plot to {save_path}")
    plt.close()


def print_comparison_table(results: List[CommunicationResult]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print(f"{'Strategy':<25} {'Comms':>8} {'Ratio':>8} {'Mean(m)':>10} {'Max(m)':>10} {'P95(m)':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r.strategy:<25} {r.n_communications:>8} {r.communication_ratio:>8.3f} "
            f"{r.mean_error:>10.2f} {r.max_error:>10.2f} {r.p95_error:>10.2f}"
        )
    print("=" * 80)
