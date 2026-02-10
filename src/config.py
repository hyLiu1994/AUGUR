"""
Experiment configuration management.

Single source of truth for all hyperparameters. Supports:
- YAML config files (configs/*.yaml)
- CLI overrides (--key value)
- Programmatic construction
"""

import os
import argparse
from dataclasses import dataclass, fields, asdict
from datetime import datetime
from typing import Optional

import yaml


@dataclass
class ExperimentConfig:
    # --- Data ---
    data_dir: str = ""
    dataset_type: Optional[str] = None  # auto-detect if None
    data_fraction: float = 1.0
    min_traj_len: int = 50
    max_test_trajs: int = 500

    # --- Model ---
    model_type: str = "mdn"  # mcdropout | heteroscedastic | mdn
    hidden_dim: int = 128
    n_components: int = 3
    dropout: float = 0.2
    seq_len: int = 20
    pred_len: int = 1

    # --- Training ---
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    ss_max: float = 0.5  # Scheduled Sampling max ratio (0 = disabled)
    load_model: Optional[str] = None

    # --- Simulation ---
    mc_samples: int = 30
    epsilon_values: str = "10,20,30,50,100"  # comma-separated
    strategies: str = "all"  # comma-separated, e.g. "dead_reckoning,proactive_norm" or "all"

    # --- Output ---
    results_dir: str = "results"
    csv_log: str = "results/experiment_log.csv"
    note: str = ""
    seed: int = 42

    @property
    def epsilon_list(self):
        return [float(x) for x in self.epsilon_values.split(",")]

    @property
    def strategies_list(self):
        if self.strategies.strip().lower() == "all":
            return None  # None means run all
        return [s.strip() for s in self.strategies.split(",")]


def load_config(yaml_path: Optional[str] = None, cli_args: Optional[list] = None) -> ExperimentConfig:
    """
    Load config with priority: CLI args > YAML file > defaults.

    Usage:
        config = load_config()                              # defaults + CLI
        config = load_config("configs/geolife_mdn.yaml")   # YAML + CLI
    """
    # Start with defaults
    config_dict = {}

    # Layer 1: YAML file
    if yaml_path is None:
        # Check CLI for --config
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=str, default=None)
        pre_args, _ = parser.parse_known_args(cli_args)
        yaml_path = pre_args.config

    if yaml_path and os.path.exists(yaml_path):
        config_dict = _load_yaml_with_base(yaml_path)

    # Layer 2: CLI overrides
    parser = _build_argparse()
    cli_parsed, _ = parser.parse_known_args(cli_args)
    cli_dict = {k: v for k, v in vars(cli_parsed).items()
                if k != "config" and v is not None}
    config_dict.update(cli_dict)

    # Remove unknown keys
    valid_keys = {f.name for f in fields(ExperimentConfig)}
    config_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

    config = ExperimentConfig(**config_dict)
    # Expand ~ in paths
    config.data_dir = os.path.expanduser(config.data_dir)
    return config


def _load_yaml_with_base(yaml_path: str) -> dict:
    """Load YAML, resolving _base_ inheritance."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    base_path = data.pop("_base_", None)
    if base_path:
        base_dir = os.path.dirname(yaml_path)
        base_full = os.path.join(base_dir, base_path)
        base_data = _load_yaml_with_base(base_full)
        base_data.update(data)
        return base_data

    return data


def _build_argparse() -> argparse.ArgumentParser:
    """Auto-generate argparse from ExperimentConfig fields."""
    parser = argparse.ArgumentParser(description="AUGUR Experiment")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")

    for f in fields(ExperimentConfig):
        arg_name = f"--{f.name}"
        if f.type is bool:
            parser.add_argument(arg_name, action="store_true", default=None)
        elif f.type is Optional[int] or f.type is Optional[str]:
            base_type = str if "str" in str(f.type) else int
            parser.add_argument(arg_name, type=base_type, default=None)
        else:
            parser.add_argument(arg_name, type=f.type, default=None)

    return parser


def save_config(config: ExperimentConfig, path: str):
    """Save config to YAML."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = asdict(config)
    # Convert None to null-friendly format
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False)


def create_run_dir(config: ExperimentConfig) -> str:
    """Create timestamped results directory for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [timestamp, config.model_type]
    if config.note:
        # sanitize note for directory name
        safe_note = config.note.replace(" ", "_").replace("/", "_")[:30]
        parts.append(safe_note)
    run_name = "_".join(parts)
    run_dir = os.path.join(config.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
