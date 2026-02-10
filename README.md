# AUGUR — Proactive Uncertainty-Guided Updates for Moving Objects

**AUGUR** (pro**A**ctive **U**ncertainty-**G**uided **U**pdate f**R**amework) reduces communication overhead in moving object tracking by using prediction uncertainty to trigger updates *before* errors accumulate.

## Core Idea

Traditional MOD methods are **reactive**: they wait until prediction error exceeds ε before transmitting. AUGUR is **proactive**: it estimates *when* the prediction will fail (via uncertainty) and communicates early, achieving lower tracking error under the same communication budget.

## Project Structure

```
Code Repository/
├── configs/
│   ├── default.yaml                # Default hyperparameters (nested structure)
│   ├── geolife.yaml                # GeoLife dataset (all strategies)
│   ├── geolife_dead_reckoning.yaml # GeoLife + Dead Reckoning only
│   ├── geolife_kalman_dps.yaml     # GeoLife + Kalman DPS only
│   └── porto.yaml                  # Porto dataset (all strategies)
├── src/
│   ├── config.py               # Nested config: Data + Model + Training + Simulation + Output
│   ├── data_loader.py           # Multi-dataset: GeoLife, Porto, DiDi, T-Drive
│   ├── model.py                 # 3 models: MC Dropout, Heteroscedastic, MDN
│   ├── strategies.py            # Communication decision functions
│   ├── trainer.py               # Model training, evaluation, checkpointing
│   ├── simulator.py             # Rolling dual-prediction simulation engine
│   └── evaluate.py              # Metrics, Pareto plots, calibration plots
├── experiments/
│   └── run_experiment.py        # Main experiment entry point
├── results/{timestamp}_{model}_{strategies}/ # Versioned experiment outputs
└── README.md
```

## Configuration

Config uses **nested dataclasses** reflecting `Method = Model + Strategy`:

```yaml
# configs/default.yaml
data:
  fraction: 1.0
  min_traj_len: 50
  max_traj_len: 200

model:
  type: mdn                  # mcdropout | heteroscedastic | mdn
  hidden_dim: 128
  seq_len: 20

training:
  epochs: 50
  lr: 0.001
  ss_max: 0.5                # Scheduled Sampling

simulation:
  epsilon_values: "10,20,30,50,100"
  strategies: all

output:
  results_dir: results

seed: 42
```

CLI supports **dot notation** and **flat aliases** (backward compatible):
```bash
# These are equivalent:
--model.type mdn       vs  --model_type mdn
--training.epochs 30   vs  --epochs 30
--data.fraction 0.1    vs  --data_fraction 0.1
```

## Models

| Model | Uncertainty | Passes | Key Feature |
|-------|------------|--------|-------------|
| `TrajectoryLSTM` | MC Dropout | 30 | Baseline, multiple forward passes |
| `HeteroscedasticLSTM` | Direct variance | 1 | Aleatoric only |
| `MDNTrajectoryLSTM` | Mixture density | 1 | Multi-modal, Law of Total Variance |

All share `predict_with_uncertainty() → (mean, std_per_dim)` API, shape `(B, P, 2)`.

## Communication Strategies

### Baselines
- **Periodic**: transmit every N steps
- **Dead Reckoning (Wolfson 1999)**: linear velocity extrapolation + threshold
- **Kalman DPS**: Kalman Filter dual prediction (constant velocity model)
- **Threshold**: LSTM prediction + threshold (classic dual prediction)

### AUGUR Strategies
- **Proactive**: communicate when `pred_error + uncertainty > ε`
- **Proactive Normalized**: `pred_error + ε·u/(u+ε) > ε` (no extra hyperparameters)
- **Dynamic ε**: ε shifts based on uncertainty vs calibrated median

## Datasets

| Dataset | Sampling | Region | Status |
|---------|----------|--------|--------|
| **GeoLife** | 1-5s | Beijing | Primary |
| **Porto Taxi** | 15s | Portugal | Generalization |
| DiDi Chengdu | 2-4s | Chengdu | Requires GAIA access |
| ~~T-Drive~~ | ~5min | Beijing | Abandoned (too sparse) |

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

### Per-strategy config files (recommended)
```bash
# Dead Reckoning baseline
python experiments/run_experiment.py --config configs/geolife_dead_reckoning.yaml

# Kalman DPS baseline
python experiments/run_experiment.py --config configs/geolife_kalman_dps.yaml
```

### Full experiment (train + simulate all strategies)
```bash
python experiments/run_experiment.py --config configs/geolife.yaml
```

### Load existing model
```bash
python experiments/run_experiment.py --config configs/geolife.yaml \
    --load_model path/to/model.pt --data.fraction 0.1
```

### Run specific strategies via CLI
```bash
# Only baselines
python experiments/run_experiment.py --config configs/geolife.yaml \
    --strategies "periodic,dead_reckoning,threshold"

# Only our methods
python experiments/run_experiment.py --config configs/geolife.yaml \
    --strategies "proactive_norm,dynamic"
```

### Override parameters via CLI
```bash
python experiments/run_experiment.py --config configs/geolife.yaml \
    --training.epochs 30 --simulation.epsilon_values "50,100,200"

# Flat aliases also work:
python experiments/run_experiment.py --config configs/geolife.yaml \
    --epochs 30 --epsilon_values "50,100,200"
```

### Available Strategy Groups

| Name | Aliases | Type | Model needed |
|------|---------|------|:---:|
| `periodic` | — | Baseline | ✅ |
| `dead_reckoning` | — | Baseline | ❌ |
| `kalman_dps` | — | Baseline | ❌ |
| `threshold` | — | Baseline | ✅ |
| `proactive` | — | AUGUR | ✅ |
| `proactive_norm` | `pronorm` | AUGUR | ✅ |
| `dynamic_eps` | `dynamic` | AUGUR | ✅ |
| `unc_accum` | — | AUGUR | ✅ |
| `unc_decay` | — | AUGUR | ✅ |

## Experimental Baselines (Planned)

| Baseline | Type | Status |
|----------|------|--------|
| Periodic | Trivial | ✅ Implemented |
| Dead Reckoning (Wolfson) | Location Update | ✅ Implemented |
| Threshold (LSTM) | Dual Prediction | ✅ Implemented |
| Kalman DPS | Dual Prediction | ✅ Implemented |
| LSTM-DPS (no unc) | Dual Prediction | ✅ = Threshold |
| Chen STSR | Safe Region | 🔧 TODO |
| GRTS (Lange) | Tracking Protocol | 🔧 TODO |
| U-OTPC | Uncertainty-driven | 🔧 TODO |

## Known Issues

- **Error compounding in LSTM dual prediction**: server's input buffer accumulates prediction errors between communications. On communication, only the current step is corrected but the preceding seq_len-1 steps remain polluted. This causes LSTM threshold to communicate *more* than Dead Reckoning at the same ε. Mitigation: Scheduled Sampling during training (`training.ss_max` parameter).

## Citation

```bibtex
@article{augur2026,
  title={AUGUR: Proactive Uncertainty-Guided Updates for Moving Objects},
  author={Liu, Hengyu and Li, Tianyi and Torp, Kristian and Li, Yushuai and Pedersen, Torben Bach and Jensen, Christian S.},
  journal={PVLDB},
  year={2026}
}
```
