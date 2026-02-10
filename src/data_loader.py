"""
Trajectory data loader and preprocessor.

Supports multiple GPS trajectory datasets:
- GeoLife: 182 users, 3+ years (Apr 2007 - Aug 2012), 1-5s intervals
- Porto Taxi: 442 taxis, 1 year (Jul 2013 - Jun 2014), 15s intervals
- DiDi Chengdu: 14,000+ vehicles, Nov 2016, 2-4s intervals
- T-Drive: ~10,000 Beijing taxis, one week (Feb 2-8, 2008), ~300s intervals

All loaders produce a unified DataFrame: [taxi_id, timestamp, longitude, latitude]
The rest of the pipeline (clean → segment → create_sequences) is shared.
"""

import os
import json
import glob
import hashlib
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from torch.utils.data import Dataset, DataLoader


# Bounding boxes for different cities/datasets
BOUNDS = {
    "beijing_wide": {  # T-Drive: wider Beijing area
        "lon_min": 115.5, "lon_max": 117.5,
        "lat_min": 39.4, "lat_max": 41.0,
    },
    "beijing": {  # GeoLife: core Beijing area (commonly used in literature)
        "lon_min": 116.1, "lon_max": 116.7,
        "lat_min": 39.7, "lat_max": 40.1,
    },
    "porto": {  # Porto, Portugal
        "lon_min": -8.735, "lon_max": -8.520,
        "lat_min": 41.100, "lat_max": 41.250,
    },
    "chengdu": {  # Chengdu, China (DiDi dataset)
        "lon_min": 103.9, "lon_max": 104.2,
        "lat_min": 30.55, "lat_max": 30.80,
    },
}

# Legacy alias
BEIJING_BOUNDS = BOUNDS["beijing_wide"]


def load_tdrive_raw(data_dir: str, max_taxis: Optional[int] = None) -> pd.DataFrame:
    """
    Load raw T-Drive trajectory files into a single DataFrame.

    Args:
        data_dir: Path to directory containing .txt files
        max_taxis: If set, only load this many taxi files (for quick testing)

    Returns:
        DataFrame with columns: [taxi_id, timestamp, longitude, latitude]
    """
    txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if max_taxis is not None:
        txt_files = txt_files[:max_taxis]

    dfs = []
    for f in txt_files:
        try:
            df = pd.read_csv(
                f,
                header=None,
                names=["taxi_id", "timestamp", "longitude", "latitude"],
                parse_dates=["timestamp"],
            )
            dfs.append(df)
        except Exception:
            continue  # skip malformed files

    data = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(txt_files)} files, {len(data)} GPS points")
    return data


def load_geolife_raw(data_dir: str, max_users: Optional[int] = None) -> pd.DataFrame:
    """
    Load raw GeoLife trajectory files into a single DataFrame.

    GeoLife PLT format (per file):
    - Lines 1-6: header (skip)
    - Line 7+: lat, lon, 0, altitude, days_since_1899, date, time

    Directory structure: data_dir/Data/000/Trajectory/*.plt

    Args:
        data_dir: Path to GeoLife root (containing Data/ folder)
        max_users: If set, only load this many users (for quick testing)

    Returns:
        DataFrame with columns: [taxi_id, timestamp, longitude, latitude]
        (taxi_id = user_id for consistency with T-Drive interface)
    """
    data_root = os.path.join(data_dir, "Data")
    if not os.path.isdir(data_root):
        data_root = data_dir  # fallback: user pointed directly at Data/

    user_dirs = sorted([d for d in os.listdir(data_root)
                        if os.path.isdir(os.path.join(data_root, d))])
    if max_users is not None:
        user_dirs = user_dirs[:max_users]

    dfs = []
    n_files = 0
    for user_id in user_dirs:
        traj_dir = os.path.join(data_root, user_id, "Trajectory")
        if not os.path.isdir(traj_dir):
            continue

        plt_files = sorted(glob.glob(os.path.join(traj_dir, "*.plt")))
        for f in plt_files:
            try:
                df = pd.read_csv(
                    f,
                    skiprows=6,
                    header=None,
                    names=["latitude", "longitude", "zero", "altitude",
                           "days", "date", "time"],
                )
                df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
                df["taxi_id"] = user_id  # reuse taxi_id column name
                df = df[["taxi_id", "timestamp", "longitude", "latitude"]]
                dfs.append(df)
                n_files += 1
            except Exception:
                continue

    data = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(user_dirs)} users, {n_files} files, {len(data)} GPS points")
    return data


def load_porto_raw(data_dir: str, max_taxis: Optional[int] = None) -> pd.DataFrame:
    """
    Load Porto Taxi dataset (ECML/PKDD 2015).

    Format: Single CSV file (train.csv) with POLYLINE column containing
    JSON-encoded GPS coordinates [[lon,lat], ...] at 15-second intervals.

    Args:
        data_dir: Path to directory containing train.csv
        max_taxis: If set, only load trips from this many unique taxis

    Returns:
        DataFrame with columns: [taxi_id, timestamp, longitude, latitude]
    """
    csv_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Porto dataset not found: {csv_path}")

    # Read CSV — POLYLINE is a JSON string, parse it lazily
    print(f"Reading Porto CSV (this may take a minute)...")
    df_raw = pd.read_csv(csv_path, usecols=["TRIP_ID", "TAXI_ID", "TIMESTAMP",
                                              "MISSING_DATA", "POLYLINE"])

    # Filter out trips with missing data or empty polylines
    df_raw = df_raw[df_raw["MISSING_DATA"] == False]  # noqa: E712
    df_raw = df_raw[df_raw["POLYLINE"].str.len() > 4]  # skip "[]" and empty

    # Optional: limit taxis
    if max_taxis is not None:
        unique_taxis = df_raw["TAXI_ID"].unique()[:max_taxis]
        df_raw = df_raw[df_raw["TAXI_ID"].isin(unique_taxis)]

    print(f"  Parsing {len(df_raw)} trips...")

    # Vectorized parsing: much faster than iterrows
    all_taxi_ids = []
    all_timestamps = []
    all_lons = []
    all_lats = []
    n_trips = 0

    for taxi_id, timestamp, polyline_str in zip(
        df_raw["TAXI_ID"].values, df_raw["TIMESTAMP"].values, df_raw["POLYLINE"].values
    ):
        try:
            polyline = json.loads(polyline_str)
        except (json.JSONDecodeError, TypeError):
            continue
        if len(polyline) < 2:
            continue

        n_pts = len(polyline)
        coords = np.array(polyline)  # (n_pts, 2) as [lon, lat]
        taxi_str = str(taxi_id)

        all_taxi_ids.extend([taxi_str] * n_pts)
        all_timestamps.extend(
            [pd.Timestamp(int(timestamp) + i * 15, unit="s") for i in range(n_pts)]
        )
        all_lons.extend(coords[:, 0].tolist())
        all_lats.extend(coords[:, 1].tolist())
        n_trips += 1

    data = pd.DataFrame({
        "taxi_id": all_taxi_ids,
        "timestamp": all_timestamps,
        "longitude": all_lons,
        "latitude": all_lats,
    })
    print(f"Loaded {n_trips} trips, {len(data)} GPS points")
    return data


def load_didi_raw(data_dir: str, max_taxis: Optional[int] = None) -> pd.DataFrame:
    """
    Load DiDi Chengdu GAIA dataset.

    Format: CSV files (one per day), no header.
    Columns: driver_id, order_id, timestamp, longitude, latitude

    Directory structure: data_dir/*.csv or data_dir/gps_*.csv

    Args:
        data_dir: Path to directory containing CSV files
        max_taxis: If set, only load data from this many unique drivers

    Returns:
        DataFrame with columns: [taxi_id, timestamp, longitude, latitude]
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        # Try subdirectory patterns
        csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(
                f,
                header=None,
                names=["driver_id", "order_id", "timestamp", "longitude", "latitude"],
            )
            # Convert Unix timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df["taxi_id"] = df["driver_id"].astype(str)
            df = df[["taxi_id", "timestamp", "longitude", "latitude"]]
            dfs.append(df)
        except Exception:
            continue

    data = pd.concat(dfs, ignore_index=True)

    if max_taxis is not None:
        unique_ids = data["taxi_id"].unique()[:max_taxis]
        data = data[data["taxi_id"].isin(unique_ids)]

    print(f"Loaded {len(csv_files)} files, {len(data)} GPS points")
    return data


def clean_trajectories(df: pd.DataFrame, bounds: Optional[dict] = None) -> pd.DataFrame:
    """
    Clean raw GPS data:
    1. Remove points outside bounding box (default: Beijing)
    2. Remove duplicate timestamps per taxi
    3. Sort by taxi_id and timestamp
    """
    n_before = len(df)

    if bounds is None:
        bounds = BEIJING_BOUNDS

    # Filter by bounding box
    mask = (
        (df["longitude"] >= bounds["lon_min"])
        & (df["longitude"] <= bounds["lon_max"])
        & (df["latitude"] >= bounds["lat_min"])
        & (df["latitude"] <= bounds["lat_max"])
    )
    df = df[mask].copy()

    # Remove duplicate timestamps per taxi
    df = df.drop_duplicates(subset=["taxi_id", "timestamp"])

    # Sort
    df = df.sort_values(["taxi_id", "timestamp"]).reset_index(drop=True)

    print(f"Cleaned: {n_before} -> {len(df)} points ({n_before - len(df)} removed)")
    return df


def segment_trajectories(
    df: pd.DataFrame,
    max_gap_seconds: int = 600,
    min_segment_len: int = 30,
    max_speed_mps: float = 42.0,  # 150 km/h; overridden per-dataset in load_and_segment()
) -> List[pd.DataFrame]:
    """
    Segment continuous trajectories by splitting at:
    1. Large time gaps (> max_gap_seconds)
    2. Implausible speed jumps (> max_speed_mps)

    This approach preserves valid sub-segments instead of discarding
    entire trajectories when a single GPS jump occurs.

    Args:
        df: Cleaned trajectory DataFrame
        max_gap_seconds: Max allowed gap (seconds) before splitting.
            T-Drive median interval is ~300s, so 600s is a reasonable cutoff.
        min_segment_len: Minimum number of points in a segment.
        max_speed_mps: Max plausible speed in m/s. Steps exceeding this
            are treated as GPS jumps and used as split points.

    Returns:
        List of trajectory segments (each a DataFrame)
    """
    segments = []
    n_time_splits = 0
    n_speed_splits = 0

    for taxi_id, group in df.groupby("taxi_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < 2:
            continue

        # Compute time gaps
        time_diff = group["timestamp"].diff().dt.total_seconds()

        # Compute spatial distances between consecutive points
        lats = group["latitude"].values.astype(np.float64)
        lons = group["longitude"].values.astype(np.float64)
        lat_rad = np.radians(lats[:-1])
        dlat = np.diff(lats) * 110540  # meters North
        dlon = np.diff(lons) * 111320 * np.cos(lat_rad)  # meters East
        distances = np.sqrt(dlat**2 + dlon**2)  # meters

        # Compute speeds (m/s) — need time_diff for consecutive pairs
        dt = time_diff.values[1:]  # skip first NaN
        # Avoid division by zero for duplicate timestamps
        dt_safe = np.maximum(dt, 1.0)
        speeds = distances / dt_safe

        # Find split points: time gaps OR speed jumps
        split_mask = np.zeros(len(group), dtype=bool)
        # Time gap splits (index in group)
        time_split_idx = time_diff[time_diff > max_gap_seconds].index.tolist()
        for idx in time_split_idx:
            split_mask[idx] = True
        n_time_splits += len(time_split_idx)

        # Speed jump splits (speeds array is offset by 1 from group index)
        speed_jump_idx = np.where(speeds > max_speed_mps)[0] + 1  # +1 for group index offset
        for idx in speed_jump_idx:
            if idx < len(split_mask):
                split_mask[idx] = True
        n_speed_splits += len(speed_jump_idx)

        # Split at marked points
        split_indices = np.where(split_mask)[0].tolist()
        prev_idx = 0
        for idx in split_indices:
            seg = group.iloc[prev_idx:idx]
            if len(seg) >= min_segment_len:
                segments.append(seg)
            prev_idx = idx
        # Last segment
        seg = group.iloc[prev_idx:]
        if len(seg) >= min_segment_len:
            segments.append(seg)

    print(f"Segmented into {len(segments)} trajectory segments")
    print(f"  Split reasons: {n_time_splits} time gaps, {n_speed_splits} speed jumps (>{max_speed_mps:.0f} m/s = {max_speed_mps*3.6:.0f} km/h)")
    if segments:
        lengths = [len(s) for s in segments]
        print(f"  Segment lengths: min={min(lengths)}, median={int(np.median(lengths))}, "
              f"max={max(lengths)}, mean={np.mean(lengths):.0f}")
    return segments


def _cache_path(data_dir: str, max_taxis: Optional[int], max_gap: int,
                min_seg: int, max_speed: float, bounds: Optional[dict] = None) -> str:
    """Generate a deterministic cache file path based on processing parameters."""
    bounds_str = ""
    if bounds:
        bounds_str = f"|{bounds['lon_min']},{bounds['lon_max']},{bounds['lat_min']},{bounds['lat_max']}"
    key = f"{data_dir}|{max_taxis}|{max_gap}|{min_seg}|{max_speed:.1f}{bounds_str}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    cache_dir = os.path.join(data_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"segments_{h}.pkl")


def _detect_dataset(data_dir: str) -> str:
    """Auto-detect dataset type from directory structure."""
    # GeoLife: has Data/ subfolder with numbered user dirs containing Trajectory/*.plt
    geolife_data = os.path.join(data_dir, "Data")
    if os.path.isdir(geolife_data):
        sample_dirs = os.listdir(geolife_data)
        for d in sample_dirs[:5]:
            traj_dir = os.path.join(geolife_data, d, "Trajectory")
            if os.path.isdir(traj_dir) and glob.glob(os.path.join(traj_dir, "*.plt")):
                return "geolife"

    # Porto: has train.csv with POLYLINE column
    porto_csv = os.path.join(data_dir, "train.csv")
    if os.path.exists(porto_csv):
        # Quick check: read header to confirm it's Porto format
        try:
            header = pd.read_csv(porto_csv, nrows=0).columns.tolist()
            if "POLYLINE" in header:
                return "porto"
        except Exception:
            pass

    # DiDi: has CSV files with 5 columns (driver_id, order_id, timestamp, lon, lat)
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if csv_files:
        try:
            sample = pd.read_csv(csv_files[0], header=None, nrows=5)
            if sample.shape[1] == 5:  # 5 columns = DiDi format
                return "didi"
        except Exception:
            pass

    # T-Drive: has .txt files directly in data_dir
    if glob.glob(os.path.join(data_dir, "*.txt")):
        return "tdrive"

    raise ValueError(
        f"Cannot detect dataset type in {data_dir}. "
        f"Supported: GeoLife (Data/*/Trajectory/*.plt), Porto (train.csv with POLYLINE), "
        f"DiDi (*.csv with 5 columns), T-Drive (*.txt)"
    )


# Default parameters per dataset
# All dataset-specific config in one place for easy extension
DATASET_DEFAULTS = {
    "geolife": {
        "bounds": BOUNDS["beijing"],
        "max_gap_seconds": 30,     # 1-5s interval → 30s gap = split
        "min_segment_len": 50,
        "max_speed_mps": 42.0,     # 150 km/h
        "description": "182 users in Beijing, 1-5s sampling",
    },
    "porto": {
        "bounds": BOUNDS["porto"],
        "max_gap_seconds": 60,     # 15s interval → 60s gap = split
        "min_segment_len": 20,     # Porto trips are shorter, 20 pts = 5 min
        "max_speed_mps": 42.0,     # 150 km/h
        "description": "Porto taxis, 15s sampling",
    },
    "didi": {
        "bounds": BOUNDS["chengdu"],
        "max_gap_seconds": 30,     # 2-4s interval → 30s gap = split
        "min_segment_len": 50,
        "max_speed_mps": 42.0,     # 150 km/h
        "description": "DiDi Chengdu, 2-4s sampling",
    },
    "tdrive": {
        "bounds": BOUNDS["beijing_wide"],
        "max_gap_seconds": 600,    # 5min median interval → 10min gap = split
        "min_segment_len": 50,
        "max_speed_mps": 42.0,     # 150 km/h
        "description": "Beijing taxis, ~300s sampling",
    },
}


def load_and_segment(
    data_dir: str,
    max_taxis: Optional[int] = None,
    max_gap_seconds: Optional[int] = None,
    min_segment_len: Optional[int] = None,
    max_speed_mps: Optional[float] = None,
    dataset_type: Optional[str] = None,
    use_cache: bool = True,
) -> List[pd.DataFrame]:
    """
    Load, clean, and segment trajectory data — with disk caching.
    Auto-detects dataset type (T-Drive or GeoLife) if not specified.

    First call processes raw data and saves segments to cache.
    Subsequent calls with the same parameters load from cache.
    """
    # Auto-detect or use specified
    if dataset_type is None:
        dataset_type = _detect_dataset(data_dir)
    print(f"Dataset type: {dataset_type}")

    # Fill defaults from dataset-specific config
    defaults = DATASET_DEFAULTS[dataset_type]
    if max_gap_seconds is None:
        max_gap_seconds = defaults["max_gap_seconds"]
    if min_segment_len is None:
        min_segment_len = defaults["min_segment_len"]
    if max_speed_mps is None:
        max_speed_mps = defaults["max_speed_mps"]

    bounds = defaults.get("bounds", BEIJING_BOUNDS)
    cp = _cache_path(data_dir, max_taxis, max_gap_seconds, min_segment_len, max_speed_mps, bounds)

    if use_cache and os.path.exists(cp):
        print(f"Loading cached segments from {cp}")
        with open(cp, "rb") as f:
            segments = pickle.load(f)
        print(f"  {len(segments)} segments loaded from cache")
        if segments:
            lengths = [len(s) for s in segments]
            print(f"  Segment lengths: min={min(lengths)}, median={int(np.median(lengths))}, "
                  f"max={max(lengths)}, mean={np.mean(lengths):.0f}")
        return segments

    # Load raw data based on dataset type
    _loaders = {
        "geolife": lambda: load_geolife_raw(data_dir, max_users=max_taxis),
        "porto": lambda: load_porto_raw(data_dir, max_taxis=max_taxis),
        "didi": lambda: load_didi_raw(data_dir, max_taxis=max_taxis),
        "tdrive": lambda: load_tdrive_raw(data_dir, max_taxis=max_taxis),
    }
    if dataset_type not in _loaders:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Supported: {list(_loaders.keys())}")
    df = _loaders[dataset_type]()

    df = clean_trajectories(df, bounds=bounds)
    segments = segment_trajectories(df, max_gap_seconds, min_segment_len, max_speed_mps)

    # Save to cache
    if use_cache:
        with open(cp, "wb") as f:
            pickle.dump(segments, f)
        size_mb = os.path.getsize(cp) / 1024 / 1024
        print(f"  Cached segments to {cp} ({size_mb:.1f} MB)")

    return segments


def latlon_to_meters(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert latitude/longitude to approximate meter offsets from the first point.
    Uses simple equirectangular projection (good enough for city-scale).
    """
    lat_ref, lon_ref = lat[0], lon[0]
    lat_rad = np.radians(lat_ref)

    x = (lon - lon_ref) * 111320 * np.cos(lat_rad)  # meters East
    y = (lat - lat_ref) * 110540  # meters North
    return x, y


def create_sequences(
    segments: List[pd.DataFrame],
    seq_len: int = 20,
    pred_len: int = 5,
    stride: int = 1,
    max_displacement_per_step: float = 25200.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Create input/target sequences for trajectory prediction.

    Note: By this point, segment_trajectories() has already filtered out
    GPS jumps via speed-based splitting (>42 m/s). The displacement filter
    here is a very loose safety net. Default = 42 m/s * 600s = 25200m,
    matching the theoretical max from segmentation parameters.

    Args:
        segments: List of trajectory segments (already speed-filtered)
        seq_len: Number of historical steps as input
        pred_len: Number of future steps to predict
        stride: Sliding window stride
        max_displacement_per_step: Safety net for extreme outliers.
            Default 25200m (42 m/s * 600s — effectively disabled).

    Returns:
        inputs: (N, seq_len, 2) normalized displacement sequences
        targets: (N, pred_len, 2) normalized future displacements
        stats: dict with 'mean' and 'std' for denormalization
    """
    inputs_list = []
    targets_list = []
    n_filtered = 0

    for seg in segments:
        lats = seg["latitude"].values
        lons = seg["longitude"].values
        x, y = latlon_to_meters(lats, lons)

        # Compute displacements (velocity-like features)
        dx = np.diff(x).astype(np.float64)
        dy = np.diff(y).astype(np.float64)
        displacements = np.stack([dx, dy], axis=-1)  # (T-1, 2)

        total_len = seq_len + pred_len
        for i in range(0, len(displacements) - total_len + 1, stride):
            window = displacements[i : i + total_len]

            # Safety net: skip if any single step has displacement > threshold
            step_norms = np.sqrt((window ** 2).sum(axis=-1))
            if step_norms.max() > max_displacement_per_step:
                n_filtered += 1
                continue

            inp = window[:seq_len]
            tgt = window[seq_len:]
            inputs_list.append(inp)
            targets_list.append(tgt)

    inputs = np.array(inputs_list, dtype=np.float32)
    targets = np.array(targets_list, dtype=np.float32)

    # Compute normalization stats from inputs
    all_data = np.concatenate([inputs.reshape(-1, 2), targets.reshape(-1, 2)], axis=0)
    stats = {
        "mean": all_data.mean(axis=0).astype(np.float32),
        "std": all_data.std(axis=0).astype(np.float32),
    }
    # Prevent division by zero
    stats["std"] = np.maximum(stats["std"], 1e-6)

    # Normalize
    inputs = (inputs - stats["mean"]) / stats["std"]
    targets = (targets - stats["mean"]) / stats["std"]

    print(f"Created {len(inputs)} sequences (seq_len={seq_len}, pred_len={pred_len})")
    if n_filtered > 0:
        print(f"  Filtered {n_filtered} sequences with displacement > {max_displacement_per_step}m")
    print(f"  Normalization stats: mean={stats['mean']}, std={stats['std']}")
    return inputs, targets, stats


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory sequences."""

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def prepare_dataloaders(
    data_dir: str,
    max_taxis: Optional[int] = None,
    seq_len: int = 20,
    pred_len: int = 5,
    batch_size: int = 256,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    End-to-end pipeline: load -> clean -> segment -> sequence -> DataLoader.

    Returns:
        train_loader, val_loader, test_loader, stats (for denormalization)
    """
    # Load and preprocess (with caching)
    segments = load_and_segment(data_dir, max_taxis=max_taxis)
    inputs, targets, stats = create_sequences(segments, seq_len=seq_len, pred_len=pred_len)

    # Split
    n = len(inputs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Shuffle indices
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    inputs, targets = inputs[indices], targets[indices]

    train_ds = TrajectoryDataset(inputs[:n_train], targets[:n_train])
    val_ds = TrajectoryDataset(inputs[n_train : n_train + n_val], targets[n_train : n_train + n_val])
    test_ds = TrajectoryDataset(inputs[n_train + n_val :], targets[n_train + n_val :])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, stats
