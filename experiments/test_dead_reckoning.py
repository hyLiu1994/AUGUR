"""Quick test: Dead Reckoning only, no ML model needed."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator import rolling_simulate_dead_reckoning
from src.data_loader import load_and_segment, latlon_to_meters
from tqdm import tqdm


def prepare_trajs(data_dir, min_traj_len=50):
    segments = load_and_segment(data_dir, min_segment_len=min_traj_len)
    trajectories = []
    for seg in segments:
        lats = seg["latitude"].values
        lons = seg["longitude"].values
        x, y = latlon_to_meters(lats, lons)
        positions = np.stack([x, y], axis=-1).astype(np.float64)
        displacements = np.diff(positions, axis=0)
        trajectories.append({"positions": positions, "displacements": displacements})
    return trajectories


trajs = prepare_trajs("D:/Datasets/Geolife Trajectories 1.3")

rng = np.random.default_rng(42)
rng.shuffle(trajs)

# Match rolling_validation.py: data_fraction=0.1
n_use = max(10, int(len(trajs) * 0.1))
trajs = trajs[:n_use]

n_train = int(len(trajs) * 0.7)
test_trajs = trajs[n_train:][:500]
print(f"\nTest: {len(test_trajs)} trajectories\n")

print(f"{'Strategy':<25} {'Ratio':>8} {'Mean(m)':>10} {'Max(m)':>10} {'P95(m)':>10}")
print("-" * 65)

for eps in [10, 20, 30, 50, 100]:
    errs, comms, steps = [], 0, 0
    for t in tqdm(test_trajs, desc=f"DR {eps}m", leave=False):
        r = rolling_simulate_dead_reckoning(t, epsilon=eps)
        errs.append(r["errors"])
        comms += r["n_comms"]
        steps += r["n_steps"]
    e = np.concatenate(errs)
    print(f"dead_reckoning_{eps}m{'':<6} {comms/steps:>8.3f} {e.mean():>10.2f} {e.max():>10.2f} {np.percentile(e,95):>10.2f}")
