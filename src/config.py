"""
Experiment configuration management.

Single source of truth for all hyperparameters. Supports:
- YAML config files with nested structure (configs/*.yaml)
- CLI overrides with dot notation (--model.type mdn) or flat aliases (--model_type mdn)
- Programmatic construction

Structure: ExperimentConfig = DataConfig + ModelConfig + TrainingConfig + SimulationConfig + OutputConfig + seed
"""

import os
import sys
import argparse
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime
from typing import Optional, List

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Dataset and data filtering parameters."""
    dir: str = ""
    dataset_type: Optional[str] = None  # auto-detect if None
    fraction: float = 1.0
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    min_traj_len: int = 50
    max_traj_len: int = 200
    max_test_trajs: int = 500


@dataclass
class ModelConfig:
    """Model architecture parameters. Unused fields are ignored per model type."""
    type: str = "mdn"  # mcdropout | heteroscedastic | mdn
    hidden_dim: int = 128
    n_components: int = 3       # MDN only
    dropout: float = 0.2
    seq_len: int = 20           # input history window (also simulator buffer length)
    pred_len: int = 1
    mc_samples: int = 30        # MC Dropout only


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    ss_max: float = 0.5         # Scheduled Sampling max ratio (0 = disabled)
    load_model: Optional[str] = None


@dataclass
class SimulationConfig:
    """Communication simulation parameters."""
    epsilon_values: str = "10,20,30,50,100"  # comma-separated
    strategies: str = "all"  # comma-separated or "all"

    @property
    def epsilon_list(self) -> List[float]:
        return [float(x) for x in self.epsilon_values.split(",")]

    @property
    def strategies_list(self) -> Optional[List[str]]:
        if self.strategies.strip().lower() == "all":
            return None  # None means run all
        return [s.strip() for s in self.strategies.split(",")]


@dataclass
class OutputConfig:
    """Output and logging parameters."""
    results_dir: str = "results"
    csv_log: str = "results/experiment_log.csv"
    note: str = ""


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Top-level experiment configuration with nested sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 42

    # --- Convenience delegation ---
    @property
    def epsilon_list(self) -> List[float]:
        return self.simulation.epsilon_list

    @property
    def strategies_list(self) -> Optional[List[str]]:
        return self.simulation.strategies_list


# ---------------------------------------------------------------------------
# Flat CLI aliases → dotted names
# ---------------------------------------------------------------------------

_FLAT_ALIASES = {
    # Data
    "data_dir": "data.dir",
    "dataset_type": "data.dataset_type",
    "data_fraction": "data.fraction",
    "min_traj_len": "data.min_traj_len",
    "max_traj_len": "data.max_traj_len",
    "max_test_trajs": "data.max_test_trajs",
    "train_ratio": "data.train_ratio",
    "val_ratio": "data.val_ratio",
    # Model
    "model_type": "model.type",
    "hidden_dim": "model.hidden_dim",
    "n_components": "model.n_components",
    "dropout": "model.dropout",
    "seq_len": "model.seq_len",
    "pred_len": "model.pred_len",
    "mc_samples": "model.mc_samples",
    # Training
    "epochs": "training.epochs",
    "batch_size": "training.batch_size",
    "lr": "training.lr",
    "ss_max": "training.ss_max",
    "load_model": "training.load_model",
    # Simulation
    "epsilon_values": "simulation.epsilon_values",
    "strategies": "simulation.strategies",
    # Output
    "results_dir": "output.results_dir",
    "csv_log": "output.csv_log",
    "note": "output.note",
}


# ---------------------------------------------------------------------------
# YAML loading with deep merge
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge: override values win, dicts merge recursively."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_yaml_with_base(yaml_path: str) -> dict:
    """Load YAML, resolving _base_ inheritance with deep merge."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    base_path = data.pop("_base_", None)
    if base_path:
        base_dir = os.path.dirname(yaml_path)
        base_full = os.path.join(base_dir, base_path)
        base_data = _load_yaml_with_base(base_full)
        return _deep_merge(base_data, data)

    return data


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _add_field_arg(parser: argparse.ArgumentParser, arg_name: str, f):
    """Add a single dataclass field as an argparse argument."""
    if f.type is bool:
        parser.add_argument(arg_name, action="store_true", default=None)
    elif f.type is Optional[int] or f.type is Optional[str]:
        base_type = str if "str" in str(f.type) else int
        parser.add_argument(arg_name, type=base_type, default=None)
    else:
        parser.add_argument(arg_name, type=f.type, default=None)


def _build_argparse() -> argparse.ArgumentParser:
    """Build argparse with dotted names + flat aliases for all config fields."""
    parser = argparse.ArgumentParser(description="AUGUR Experiment")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")

    # Nested fields with dot notation
    _SUB_CONFIGS = {
        "data": DataConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "simulation": SimulationConfig,
        "output": OutputConfig,
    }

    for prefix, cls in _SUB_CONFIGS.items():
        for f in fields(cls):
            dotted = f"--{prefix}.{f.name}"
            _add_field_arg(parser, dotted, f)

    # Top-level
    parser.add_argument("--seed", type=int, default=None)

    # Flat aliases (backward compat)
    for flat_name, dotted_name in _FLAT_ALIASES.items():
        # Find the field type from sub-config
        prefix, fname = dotted_name.split(".")
        cls = _SUB_CONFIGS[prefix]
        matching = [f for f in fields(cls) if f.name == fname]
        if matching:
            _add_field_arg(parser, f"--{flat_name}", matching[0])

    return parser


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

def _dict_to_config(d: dict) -> ExperimentConfig:
    """Construct ExperimentConfig from a nested dict."""
    def _pick(cls, sub_dict):
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in sub_dict.items() if k in valid})

    config = ExperimentConfig(
        data=_pick(DataConfig, d.get("data", {})),
        model=_pick(ModelConfig, d.get("model", {})),
        training=_pick(TrainingConfig, d.get("training", {})),
        simulation=_pick(SimulationConfig, d.get("simulation", {})),
        output=_pick(OutputConfig, d.get("output", {})),
        seed=d.get("seed", 42),
    )
    # Expand ~ in paths
    config.data.dir = os.path.expanduser(config.data.dir)
    if config.training.load_model:
        config.training.load_model = os.path.expanduser(config.training.load_model)
    return config


def load_config(yaml_path: Optional[str] = None, cli_args: Optional[list] = None) -> ExperimentConfig:
    """
    Load config with priority: CLI args > YAML file > defaults.

    Usage:
        config = load_config()                            # defaults + CLI
        config = load_config("configs/geolife.yaml")      # YAML + CLI
    """
    # Check CLI for --config
    if yaml_path is None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=str, default=None)
        pre_args, _ = parser.parse_known_args(cli_args)
        yaml_path = pre_args.config

    # Layer 1: YAML file (nested dict)
    config_dict = {}
    if yaml_path and os.path.exists(yaml_path):
        config_dict = _load_yaml_with_base(yaml_path)

    # Layer 2: CLI overrides
    parser = _build_argparse()
    cli_parsed, _ = parser.parse_known_args(cli_args)
    cli_dict = vars(cli_parsed)

    # Apply CLI values to nested dict (two passes: dotted first, then flat aliases)
    # Pass 1: dotted keys have highest priority
    dotted_set = set()  # track which dotted paths were explicitly set
    for key, value in cli_dict.items():
        if key == "config" or value is None:
            continue
        if "." in key:
            prefix, name = key.split(".", 1)
            if prefix not in config_dict:
                config_dict[prefix] = {}
            config_dict[prefix][name] = value
            dotted_set.add(f"{prefix}.{name}")
        elif key == "seed":
            config_dict["seed"] = value

    # Pass 2: flat aliases (override YAML, but not explicit dotted CLI)
    for key, value in cli_dict.items():
        if key == "config" or value is None:
            continue
        if key in _FLAT_ALIASES:
            dotted = _FLAT_ALIASES[key]
            if dotted in dotted_set:
                continue  # explicit dotted version takes priority
            prefix, name = dotted.split(".", 1)
            if prefix not in config_dict:
                config_dict[prefix] = {}
            config_dict[prefix][name] = value

    return _dict_to_config(config_dict)


# ---------------------------------------------------------------------------
# Save / Run directory
# ---------------------------------------------------------------------------

def save_config(config: ExperimentConfig, path: str):
    """Save config to YAML (nested structure)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = asdict(config)
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False)


def _needs_model_for_strategies(config: ExperimentConfig) -> bool:
    """Check if any selected strategy requires a trained model."""
    NO_MODEL = {"dead_reckoning", "grts", "kalman_dps"}
    selected = config.strategies_list
    if selected is None:
        return True
    return any(s not in NO_MODEL for s in selected)


def create_run_dir(config: ExperimentConfig) -> str:
    """Create timestamped results directory for this run.

    Format: {timestamp}_{model}_{strategies}[_{note}]
    When all strategies are model-free: {timestamp}_{strategies}[_{note}]
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [timestamp]

    # Only include model type when model is actually used
    if _needs_model_for_strategies(config):
        parts.append(config.model.type)

    # Add strategy names
    selected = config.strategies_list
    if selected is None:
        parts.append("all")
    else:
        parts.append("_".join(selected))

    if config.output.note:
        safe_note = config.output.note.replace(" ", "_").replace("/", "_")[:30]
        parts.append(safe_note)

    run_name = "_".join(parts)
    run_dir = os.path.join(config.output.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
