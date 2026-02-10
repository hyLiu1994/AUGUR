"""
Communication strategies for moving object location updates.

Each strategy is a decision function:
    decide_*(pred_error, epsilon, ...) -> bool

The simulator calls these at each timestep to decide whether to communicate.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class CommunicationResult:
    """Result of a communication simulation run."""
    strategy: str
    n_communications: int
    n_total_steps: int
    communication_ratio: float
    errors: np.ndarray
    mean_error: float
    max_error: float
    p95_error: float
    communication_indices: List[int]


def compute_tracking_error(
    true_positions: np.ndarray,
    server_positions: np.ndarray,
) -> np.ndarray:
    """L2 tracking error between true and server positions. Shape: (T,)."""
    diff = true_positions - server_positions
    return np.sqrt((diff ** 2).sum(axis=-1))


# ---------------------------------------------------------------------------
# Strategy decision functions
# ---------------------------------------------------------------------------

def decide_threshold(pred_error, epsilon, **kwargs):
    """Reactive: communicate when error exceeds epsilon."""
    return pred_error > epsilon


def decide_proactive(pred_error, epsilon, unc_meters=0.0, **kwargs):
    """Proactive: communicate when error + raw uncertainty exceeds epsilon."""
    return pred_error + unc_meters > epsilon


def decide_proactive_norm(pred_error, epsilon, unc_meters=0.0, **kwargs):
    """
    Proactive with normalized uncertainty (main AUGUR strategy).

    f(u) = ε·u/(u+ε)
    Properties: bounded [0,ε), linear for small u, saturates for large u.
    Communicate when e_t + f(u_t) > ε.
    """
    normalized_unc = epsilon * unc_meters / (unc_meters + epsilon)
    return pred_error + normalized_unc > epsilon


def decide_dynamic_eps(pred_error, epsilon, unc_meters=0.0, median_unc=0.0, **kwargs):
    """
    Dynamic epsilon: shift ε based on uncertainty relative to calibrated median.
    High uncertainty -> lower effective epsilon (communicate earlier).
    Low uncertainty -> higher effective epsilon (save bandwidth).
    Hard cap: pred_error > ε always triggers.
    """
    norm_unc = epsilon * unc_meters / (unc_meters + epsilon)
    norm_med = epsilon * median_unc / (median_unc + epsilon)
    dynamic_epsilon = epsilon + (norm_med - norm_unc)
    if pred_error > dynamic_epsilon:
        return True
    return pred_error > epsilon  # hard guarantee


def decide_unc_accum(pred_error, epsilon, unc_meters=0.0, state=None, **kwargs):
    """
    Accumulated uncertainty: sum uncertainty since last communication.
    Communicate when accumulated risk > epsilon OR error > epsilon.
    Updates state["accumulated_risk"] in place.
    """
    if state is None:
        state = {"accumulated_risk": 0.0}
    state["accumulated_risk"] += unc_meters
    if state["accumulated_risk"] > epsilon or pred_error > epsilon:
        state["accumulated_risk"] = 0.0
        return True
    return False


def decide_unc_decay(pred_error, epsilon, unc_meters=0.0, decay=0.9, state=None, **kwargs):
    """
    Decayed accumulated uncertainty: older uncertainty fades out.
    Communicate when decayed risk > epsilon OR error > epsilon.
    Updates state["accumulated_risk"] in place.
    """
    if state is None:
        state = {"accumulated_risk": 0.0}
    state["accumulated_risk"] = decay * state["accumulated_risk"] + unc_meters
    if state["accumulated_risk"] > epsilon or pred_error > epsilon:
        state["accumulated_risk"] = 0.0
        return True
    return False


def decide_periodic(t, period, **kwargs):
    """Periodic: communicate every `period` steps."""
    return t % period == 0


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES = {
    "threshold": decide_threshold,
    "proactive": decide_proactive,
    "proactive_norm": decide_proactive_norm,
    "dynamic_eps": decide_dynamic_eps,
    "unc_accum": decide_unc_accum,
    "unc_decay": decide_unc_decay,
}

# Strategies that need uncertainty estimation
UNCERTAINTY_STRATEGIES = {"proactive", "proactive_norm", "dynamic_eps", "unc_accum", "unc_decay"}
