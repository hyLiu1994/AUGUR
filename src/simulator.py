"""
Rolling simulation engine for dual-prediction communication.

Core component: simulate how a moving object and server interact
under different communication strategies.

Extracted from experiments/rolling_validation.py.
"""

import numpy as np
import torch
from tqdm import tqdm

from src.model import TrajectoryLSTM
from src.strategies import STRATEGIES, UNCERTAINTY_STRATEGIES
from src.data_loader import load_and_segment, latlon_to_meters
from src.config import ExperimentConfig


def prepare_long_trajectories(config: ExperimentConfig):
    """
    Load trajectory segments and convert to meter-space positions/displacements.

    Returns list of dicts:
        positions: (T, 2) absolute positions in meters
        displacements: (T-1, 2) step-wise displacements in meters
    """
    segments = load_and_segment(
        config.data.dir,
        min_segment_len=config.data.min_traj_len,
    )

    max_len = config.data.max_traj_len

    trajectories = []
    for seg in segments:
        lats = seg["latitude"].values
        lons = seg["longitude"].values
        x, y = latlon_to_meters(lats, lons)

        positions = np.stack([x, y], axis=-1).astype(np.float64)

        # Split long trajectories into sub-trajectories of max_len
        for start in range(0, len(positions), max_len):
            sub_pos = positions[start : start + max_len]
            if len(sub_pos) < config.data.min_traj_len:
                break  # tail too short, discard
            sub_disp = np.diff(sub_pos, axis=0)
            trajectories.append({
                "positions": sub_pos,
                "displacements": sub_disp,
            })

    print(f"Prepared {len(trajectories)} long trajectories")
    if trajectories:
        lengths = [len(t["positions"]) for t in trajectories]
        print(f"  Length stats: min={min(lengths)}, median={int(np.median(lengths))}, "
              f"max={max(lengths)}, mean={np.mean(lengths):.0f}")
    return trajectories


def split_trajectories(trajectories, config: ExperimentConfig):
    """Split trajectories into train/test, apply data_fraction and max_test_trajs."""
    rng = np.random.default_rng(config.seed)
    rng.shuffle(trajectories)

    if config.data.fraction < 1.0:
        n_use = max(10, int(len(trajectories) * config.data.fraction))
        trajectories = trajectories[:n_use]
        print(f"  Using {config.data.fraction:.0%} of data: {n_use} trajectories")

    n_total = len(trajectories)
    n_train = int(n_total * config.data.train_ratio)
    n_val = int(n_total * config.data.val_ratio)
    train_trajs = trajectories[:n_train]
    val_trajs = trajectories[n_train:n_train + n_val]
    test_trajs = trajectories[n_train + n_val:]

    if config.data.max_test_trajs and len(test_trajs) > config.data.max_test_trajs:
        test_trajs = test_trajs[:config.data.max_test_trajs]

    def _len_stats(trajs, name):
        lengths = [len(t["positions"]) for t in trajs]
        print(f"  {name}: {len(trajs)} trajectories | "
              f"len min={min(lengths)}, median={int(np.median(lengths))}, "
              f"max={max(lengths)}, mean={np.mean(lengths):.0f}")

    _len_stats(train_trajs, "Train")
    _len_stats(val_trajs, "Val")
    _len_stats(test_trajs, "Test")
    return train_trajs, val_trajs, test_trajs


# ---------------------------------------------------------------------------
# Dead Reckoning baseline (no ML model)
# ---------------------------------------------------------------------------

def rolling_simulate_dead_reckoning(trajectory, epsilon):
    """
    Dead Reckoning baseline (Wolfson et al. 1999, ADR).

    No ML model. Linear velocity extrapolation:
    - On communication: client sends (position, velocity)
    - Between communications: server extrapolates with last velocity
    - Trigger when actual deviation > epsilon
    """
    positions = trajectory["positions"]
    T = len(positions)

    server_positions = np.zeros_like(positions)
    server_positions[0] = positions[0].copy()
    comm_indices = []
    pred_errors = np.zeros(T)

    last_comm_pos = positions[0].copy()
    last_comm_t = 0
    velocity = np.zeros(2)

    if T > 1:
        velocity = positions[1] - positions[0]

    for t in range(1, T):
        dt = t - last_comm_t
        predicted_pos = last_comm_pos + velocity * dt
        server_positions[t] = predicted_pos

        true_pos = positions[t]
        pred_error = np.sqrt(((true_pos - predicted_pos) ** 2).sum())
        pred_errors[t] = pred_error

        if pred_error > epsilon:
            server_positions[t] = true_pos.copy()
            if dt > 0:
                velocity = (true_pos - last_comm_pos) / dt
            last_comm_pos = true_pos.copy()
            last_comm_t = t
            comm_indices.append(t)

    errors = np.sqrt(((positions - server_positions) ** 2).sum(axis=-1))

    return {
        "errors": errors,
        "comm_indices": comm_indices,
        "n_comms": len(comm_indices),
        "n_steps": T,
        "mean_error": errors.mean(),
        "max_error": errors.max(),
        "p95_error": np.percentile(errors, 95),
        "uncertainties": np.zeros(T),
        "pred_errors": pred_errors,
    }


# ---------------------------------------------------------------------------
# Kalman Filter DPS baseline (no ML model)
# ---------------------------------------------------------------------------

def rolling_simulate_kalman_dps(trajectory, epsilon):
    """
    Kalman Filter Dual Prediction Scheme.

    Both client and server maintain identical Kalman filters.
    State: [px, py, vx, vy] — constant velocity model.
    Client observes true positions; server only gets updates on communication.

    Advantage over DR: Kalman smooths noisy observations and maintains
    uncertainty (P matrix), giving more stable velocity estimates.
    """
    positions = trajectory["positions"]
    T = len(positions)

    # --- Kalman Filter setup (constant velocity model) ---
    # State: [px, py, vx, vy]
    dt = 1.0  # unit time step

    # State transition matrix
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=np.float64)

    # Observation matrix (only observe position)
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float64)

    # Process noise (acceleration uncertainty)
    q = 1.0  # process noise magnitude
    Q = np.array([
        [dt**3/3, 0,       dt**2/2, 0      ],
        [0,       dt**3/3, 0,       dt**2/2],
        [dt**2/2, 0,       dt,      0      ],
        [0,       dt**2/2, 0,       dt     ],
    ], dtype=np.float64) * q

    # Measurement noise — positions are ground truth, not noisy GPS
    R = np.eye(2, dtype=np.float64) * 0.1

    # --- Initialize client and server KFs identically ---
    x_client = np.array([positions[0, 0], positions[0, 1], 0.0, 0.0])
    P_client = np.eye(4, dtype=np.float64) * 100.0

    x_server = x_client.copy()
    P_server = P_client.copy()

    # Initialize velocity from first two points
    if T > 1:
        v_init = positions[1] - positions[0]
        x_client[2:] = v_init
        x_server[2:] = v_init

    server_positions = np.zeros_like(positions)
    server_positions[0] = positions[0].copy()
    comm_indices = []
    pred_errors = np.zeros(T)

    for t in range(1, T):
        # --- Predict step (both client and server) ---
        x_client_pred = F @ x_client
        P_client_pred = F @ P_client @ F.T + Q

        x_server_pred = F @ x_server
        P_server_pred = F @ P_server @ F.T + Q

        # --- Client update (always observes true position) ---
        z = positions[t]  # true observation
        y_client = z - H @ x_client_pred  # innovation
        S_client = H @ P_client_pred @ H.T + R
        K_client = P_client_pred @ H.T @ np.linalg.inv(S_client)
        x_client = x_client_pred + K_client @ y_client
        P_client = (np.eye(4) - K_client @ H) @ P_client_pred

        # --- Server prediction (no observation yet) ---
        server_pos_pred = x_server_pred[:2]
        server_positions[t] = server_pos_pred

        # --- Communication decision ---
        true_pos = positions[t]
        pred_error = np.sqrt(((true_pos - server_pos_pred) ** 2).sum())
        pred_errors[t] = pred_error

        if pred_error > epsilon:
            # Communicate: synchronize server KF to client KF state
            # This is the correct DPS — on communication both sides align
            x_server = x_client.copy()
            P_server = P_client.copy()

            server_positions[t] = true_pos.copy()
            comm_indices.append(t)
        else:
            # No communication: server keeps prediction
            x_server = x_server_pred
            P_server = P_server_pred

    errors = np.sqrt(((positions - server_positions) ** 2).sum(axis=-1))

    return {
        "errors": errors,
        "comm_indices": comm_indices,
        "n_comms": len(comm_indices),
        "n_steps": T,
        "mean_error": errors.mean(),
        "max_error": errors.max(),
        "p95_error": np.percentile(errors, 95),
        "uncertainties": np.zeros(T),
        "pred_errors": pred_errors,
    }


# ---------------------------------------------------------------------------
# Core rolling simulation (ML-based strategies)
# ---------------------------------------------------------------------------

def rolling_simulate(
    model, trajectory, stats, device, config,
    strategy_name, epsilon_override=None, median_unc=None, decay=0.9,
):
    """
    Rolling simulation on a full trajectory with dual-prediction architecture.

    - Client has ground truth -> makes communication decisions
    - Server only knows what it received via communication
    - Between communications, server uses its OWN predictions as input
    """
    positions = trajectory["positions"]
    displacements = trajectory["displacements"]
    T = len(positions)
    seq_len = config.model.seq_len
    mc_samples = config.model.mc_samples
    run_epsilon = epsilon_override if epsilon_override is not None else config.epsilon_list[0]

    # Determine if we need uncertainty for this strategy
    is_periodic = strategy_name.startswith("periodic_")
    need_uncertainty = strategy_name in UNCERTAINTY_STRATEGIES
    is_mcdropout = isinstance(model, TrajectoryLSTM)

    # Get strategy function
    if not is_periodic:
        strategy_fn = STRATEGIES[strategy_name]

    # State for stateful strategies (unc_accum, unc_decay)
    strategy_state = {"accumulated_risk": 0.0}

    # Server's displacement buffer (normalized)
    server_disp_norm = np.zeros((T - 1, 2), dtype=np.float64)
    server_positions = np.zeros_like(positions)
    server_positions[0] = positions[0].copy()
    comm_indices = []
    uncertainties = np.zeros(T)
    pred_errors = np.zeros(T)

    stats_mean = stats["mean"]
    stats_std = stats["std"]

    # Set model mode
    if need_uncertainty and is_mcdropout:
        model.train()
    else:
        model.eval()

    for t in range(1, T):
        # Build input from server's buffer
        if t - 1 < seq_len:
            available = server_disp_norm[:t - 1]
            pad_len = seq_len - len(available)
            inp = np.concatenate([np.zeros((pad_len, 2), dtype=np.float32), available], axis=0)
        else:
            inp = server_disp_norm[t - 1 - seq_len : t - 1]

        inp_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)

        unc_meters = 0.0
        with torch.no_grad():
            if need_uncertainty:
                if is_mcdropout:
                    predictions = []
                    for _ in range(mc_samples):
                        pred = model(inp_tensor)
                        predictions.append(pred)
                    predictions = torch.stack(predictions, dim=0)
                    mean_pred = predictions.mean(dim=0).cpu().numpy()[0, 0]
                    std_pred = predictions.std(dim=0).cpu().numpy()[0, 0]
                else:
                    mean_t, std_t = model.predict_with_uncertainty(inp_tensor)
                    mean_pred = mean_t.cpu().numpy()[0, 0]
                    std_pred = std_t.cpu().numpy()[0, 0]

                std_meters = std_pred * stats_std
                unc_meters = np.sqrt((std_meters ** 2).sum())
                uncertainties[t] = unc_meters
            else:
                if is_mcdropout:
                    pred = model(inp_tensor)
                    mean_pred = pred.cpu().numpy()[0, 0]
                else:
                    mean_pred = model.predict_mean(inp_tensor).cpu().numpy()[0, 0]

        # Denormalize prediction
        pred_disp_meters = mean_pred * stats_std + stats_mean

        # Server predicts position
        predicted_pos = server_positions[t - 1] + pred_disp_meters
        true_pos = positions[t]
        server_positions[t] = predicted_pos

        pred_error = np.sqrt(((true_pos - predicted_pos) ** 2).sum())
        pred_errors[t] = pred_error

        # Communication decision
        if is_periodic:
            period = int(strategy_name.split("_")[1])
            communicated = t % period == 0
        else:
            communicated = strategy_fn(
                pred_error=pred_error,
                epsilon=run_epsilon,
                unc_meters=unc_meters,
                median_unc=median_unc or 0.0,
                decay=decay,
                state=strategy_state,
            )

        if communicated:
            server_positions[t] = true_pos.copy()
            server_disp_norm[t - 1] = (displacements[t - 1] - stats_mean) / stats_std
            comm_indices.append(t)
            strategy_state["accumulated_risk"] = 0.0
        else:
            server_disp_norm[t - 1] = mean_pred

    errors = np.sqrt(((positions - server_positions) ** 2).sum(axis=-1))

    return {
        "errors": errors,
        "comm_indices": comm_indices,
        "n_comms": len(comm_indices),
        "n_steps": T,
        "mean_error": errors.mean(),
        "max_error": errors.max(),
        "p95_error": np.percentile(errors, 95),
        "uncertainties": uncertainties,
        "pred_errors": pred_errors,
    }


# ---------------------------------------------------------------------------
# Uncertainty profiling
# ---------------------------------------------------------------------------

def profile_uncertainty(model, train_trajs, stats, config, device, n_trajs=30):
    """Profile uncertainty distribution on training trajectories."""
    print("\n=== Profiling uncertainty distribution ===")
    calib_uncs = []
    for traj in tqdm(train_trajs[:n_trajs], desc="  Profiling", leave=False):
        result = rolling_simulate(
            model, traj, stats, device, config,
            strategy_name="unc_accum",
            epsilon_override=1e6,  # high epsilon = never trigger, just collect unc
        )
        calib_uncs.append(result["uncertainties"])

    all_calib_unc = np.concatenate(calib_uncs)
    all_calib_unc = all_calib_unc[all_calib_unc > 0]
    median_unc = float(np.median(all_calib_unc))

    print(f"  Uncertainty distribution: mean={all_calib_unc.mean():.2f}m, "
          f"median={median_unc:.2f}m, "
          f"p75={np.percentile(all_calib_unc, 75):.2f}m, "
          f"p90={np.percentile(all_calib_unc, 90):.2f}m")

    return median_unc


# ---------------------------------------------------------------------------
# Run all strategies
# ---------------------------------------------------------------------------

def _want(selected, prefix):
    """Check if a strategy group should run based on --strategies filter."""
    if selected is None:  # "all"
        return True
    return any(s == prefix or s.startswith(prefix) for s in selected)


def build_strategy_configs(config: ExperimentConfig, median_unc: float):
    """
    Build the strategy parameter dict for all strategies to run.

    Respects config.strategies_list:
        "all"                              → everything
        "dead_reckoning,proactive_norm"    → only those two groups
        "threshold,pronorm"               → threshold + pronorm (alias for proactive_norm)

    Strategy group names:
        periodic, dead_reckoning, threshold,
        proactive, proactive_norm (alias: pronorm),
        dynamic_eps (alias: dynamic),
        unc_accum, unc_decay
    """
    selected = config.strategies_list  # None means all
    strategies = {}

    # Periodic baselines
    if _want(selected, "periodic"):
        for period in [5, 10, 20]:
            strategies[f"periodic_{period}"] = {
                "strategy_name": f"periodic_{period}",
            }

    # Threshold baselines
    if _want(selected, "threshold"):
        for eps in config.epsilon_list:
            strategies[f"threshold_{eps:.0f}m"] = {
                "strategy_name": "threshold",
                "epsilon_override": eps,
            }

    # Proactive
    if _want(selected, "proactive"):
        for eps in config.epsilon_list:
            strategies[f"proactive_{eps:.0f}m"] = {
                "strategy_name": "proactive",
                "epsilon_override": eps,
            }

    # Proactive Normalized (accept both "proactive_norm" and "pronorm")
    if _want(selected, "proactive_norm") or _want(selected, "pronorm"):
        for eps in config.epsilon_list:
            strategies[f"pronorm_{eps:.0f}m"] = {
                "strategy_name": "proactive_norm",
                "epsilon_override": eps,
            }

    # Dynamic epsilon (accept both "dynamic_eps" and "dynamic")
    if _want(selected, "dynamic_eps") or _want(selected, "dynamic"):
        for eps in config.epsilon_list:
            strategies[f"dynamic_{eps:.0f}m"] = {
                "strategy_name": "dynamic_eps",
                "epsilon_override": eps,
                "median_unc": median_unc,
            }

    # Uncertainty accumulation
    if _want(selected, "unc_accum"):
        for eps in config.epsilon_list:
            strategies[f"unc_accum_{eps:.0f}m"] = {
                "strategy_name": "unc_accum",
                "epsilon_override": eps,
            }

    # Uncertainty decay
    if _want(selected, "unc_decay"):
        for eps in config.epsilon_list:
            strategies[f"unc_decay_{eps:.0f}m"] = {
                "strategy_name": "unc_decay",
                "epsilon_override": eps,
            }

    return strategies


def run_all_strategies(model, test_trajs, stats, config, device, median_unc):
    """Run all configured strategies and return aggregated results."""
    strategies = build_strategy_configs(config, median_unc)

    all_results = {}

    # Model-free baselines
    _MODEL_FREE = {
        "dead_reckoning": rolling_simulate_dead_reckoning,
        "kalman_dps": rolling_simulate_kalman_dps,
    }

    for baseline_name, simulate_fn in _MODEL_FREE.items():
        if not _want(config.strategies_list, baseline_name):
            continue
        for eps in config.epsilon_list:
            result_name = f"{baseline_name}_{eps:.0f}m"
            print(f"  Running {result_name}...")
            bl_errors, bl_comms, bl_steps = [], 0, 0

            for traj in tqdm(test_trajs, desc=f"    {result_name}", leave=False):
                bl_result = simulate_fn(traj, epsilon=eps)
                bl_errors.append(bl_result["errors"])
                bl_comms += bl_result["n_comms"]
                bl_steps += bl_result["n_steps"]

            bl_all_errors = np.concatenate(bl_errors)
            all_results[result_name] = {
                "n_comms": bl_comms,
                "n_steps": bl_steps,
                "comm_ratio": bl_comms / bl_steps,
                "mean_error": float(bl_all_errors.mean()),
                "max_error": float(bl_all_errors.max()),
                "p95_error": float(np.percentile(bl_all_errors, 95)),
                "uncertainties": np.zeros(1),
                "pred_errors": np.zeros(1),
            }

    # ML-based strategies
    for name, params in strategies.items():
        if name.startswith("dead_reckoning_") or name.startswith("kalman_dps_"):
            # model-free baselines already handled above
            continue

        print(f"  Running {name}...")
        total_errors, total_comms, total_steps = [], 0, 0
        total_uncertainties, total_pred_errors = [], []

        strategy_name = params["strategy_name"]
        eps_override = params.get("epsilon_override", None)
        med_unc = params.get("median_unc", None)

        for traj in tqdm(test_trajs, desc=f"    {name}", leave=False):
            result = rolling_simulate(
                model, traj, stats, device, config,
                strategy_name=strategy_name,
                epsilon_override=eps_override,
                median_unc=med_unc,
            )
            total_errors.append(result["errors"])
            total_comms += result["n_comms"]
            total_steps += result["n_steps"]
            total_uncertainties.append(result["uncertainties"])
            total_pred_errors.append(result["pred_errors"])

        all_errors = np.concatenate(total_errors)
        all_results[name] = {
            "n_comms": total_comms,
            "n_steps": total_steps,
            "comm_ratio": total_comms / total_steps,
            "mean_error": float(all_errors.mean()),
            "max_error": float(all_errors.max()),
            "p95_error": float(np.percentile(all_errors, 95)),
            "uncertainties": np.concatenate(total_uncertainties),
            "pred_errors": np.concatenate(total_pred_errors),
        }

    return all_results
