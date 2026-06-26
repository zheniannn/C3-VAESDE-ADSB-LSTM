# C3-VAESDE-ADSB-LSTM

LSTM next-step prediction model for ADS-B trajectory anomaly detection.

---

## Purpose

Trains a recurrent model to predict the next state of an ADS-B trajectory window. Normal aircraft motion is predictable; abnormal motion (sudden turns, position jumps, random velocity, stationary clutter) produces high prediction error. This per-step prediction MSE is used as a motion-plausibility score that complements the whole-window VAE score from C2.

---

## Relationship to C1 and C2

```
C1-VAESDE-ADSB-PREPROCESSING
    → normalised ENU windows  (X_train.npy, X_test.npy)
        → C2-VAESDE-ADSB-TRAINING
            → VAE reconstruction MSE + kinematic flags
        → C3-VAESDE-ADSB-LSTM  (this repo)
            → LSTM prediction MSE
```

The three detectors are complementary:

| Detector | What it catches |
|---|---|
| VAE (C2) | whole-window distribution anomalies (e.g. random walk) |
| LSTM (C3) | step-to-step transition anomalies (e.g. sudden turn, speed change) |
| Kinematic flags | hard physical rule violations (e.g. position jump, stationarity) |

Final anomaly flag: trigger if **any** of the three fires.

---

## Input data

Place these files in `data/` before running (copy from C1 output):

```bash
cp ../C1-VAESDE-ADSB-PREPROCESSING/data/X_train.npy data/
cp ../C1-VAESDE-ADSB-PREPROCESSING/data/X_test.npy data/
```

| File | Shape | Description |
|---|---|---|
| `X_train.npy` | (1 412 436, 30, 4) | Normalised train sequences |
| `X_test.npy` | (160 946, 30, 4) | Normalised test sequences |

Feature order: `[E_m, N_m, vE_mps, vN_mps]` (all normalised to zero mean / unit std on train split).

Optional (not used by training, kept for provenance):
- `normalisation_mean.csv`, `normalisation_std.csv`
- `train_sequence_metadata.csv`, `test_sequence_metadata.csv`

---

## Model objective

For each 30-step trajectory window `x`:

```
input  : x[:-1, :]  →  shape (29, 4)   — steps 0..28
target : x[1:, :]   →  shape (29, 4)   — steps 1..29
```

The LSTM predicts the next state from the current state. Per-sequence prediction MSE becomes the anomaly score: low for normal motion, high for corrupted or implausible motion.

---

## Repository structure

```
C3-VAESDE-ADSB-LSTM/
├── configs/lstm_default.yaml     experiment settings
├── scripts/
│   ├── run_train_lstm.py         train the model
│   ├── run_score_lstm.py         score sequences, compute thresholds
│   └── run_stress_test_lstm.py   detection rates across corruption types
├── src/adsb_lstm/                installable package
│   ├── config.py                 load_config, set_seed, ensure_dir
│   ├── dataset.py                SequenceDataset
│   ├── model.py                  MotionLSTM, MotionGRU, build_model
│   ├── training.py               train_one_epoch, evaluate, fit_model
│   ├── inference.py              load_checkpoint, score_sequences
│   ├── corruption.py             stress-test corruption functions
│   ├── reporting.py              save_dataframe, save_json, print_summary_table
│   └── utils.py                  get_device, count_parameters, describe_array
├── tests/test_smoke.py
├── data/                         input files (not tracked)
└── outputs/                      generated results (not tracked)
```

---

## Quick start

```bash
pip install -e .
pytest

cp ../C1-VAESDE-ADSB-PREPROCESSING/data/X_train.npy data/
cp ../C1-VAESDE-ADSB-PREPROCESSING/data/X_test.npy data/

python scripts/run_train_lstm.py --config configs/lstm_default.yaml
python scripts/run_score_lstm.py --config configs/lstm_default.yaml
python scripts/run_stress_test_lstm.py --config configs/lstm_default.yaml
```

Set `debug_mode: false` in `configs/lstm_default.yaml` to train on the full dataset.

---

## Training

```bash
python scripts/run_train_lstm.py --config configs/lstm_default.yaml
```

Outputs to `outputs/lstm/`:
- `motion_lstm.pt` — model checkpoint (weights + config + history)
- `history.csv` — per-epoch train/test loss

Default architecture: 2-layer LSTM, hidden_dim=128, 20 epochs, Adam lr=0.001.
Set `model_type: gru` in the config to switch to GRU.

---

## Scoring

```bash
python scripts/run_score_lstm.py --config configs/lstm_default.yaml
```

Outputs to `outputs/lstm/`:
- `train_prediction_errors.csv` — per-sequence MSE breakdown (total, pos, vel, final_step)
- `test_prediction_errors.csv`
- `lstm_thresholds.csv` — p90/p95/p99 thresholds from train MSE distribution + test false-flag rates

---

## Stress testing

```bash
python scripts/run_stress_test_lstm.py --config configs/lstm_default.yaml
```

Requires `lstm_thresholds.csv` from the scoring step. Tests detection rate at train p95 against:

| Corruption | What it does |
|---|---|
| clean | no corruption (baseline false-flag rate) |
| speed_scaled_1.5 / 2.0 | velocity scaled by 1.5× or 2.0× |
| random_walk_velocity | velocities replaced with Gaussian noise |
| sudden_turn_90 | 90° velocity rotation from mid-sequence |
| position_jump | +2σ step offset in E, N from mid-sequence |
| stationary_clutter | near-zero velocity and position |

Outputs `outputs/lstm/lstm_stress_summary.csv`.

---

## Interpretation of LSTM score

The LSTM prediction MSE measures **transition plausibility**: does each observed step follow naturally from the previous one? Corruptions that violate the learned motion model produce high MSE.

Expected detection profile:
- **random_walk_velocity** — high LSTM MSE (unpredictable steps)
- **sudden_turn_90** — elevated MSE at the turn point
- **speed_scaled** — moderate MSE (velocity magnitude changed)
- **position_jump** — moderate LSTM MSE (discontinuity in position); primary detector is kinematic pv_max flag
- **stationary_clutter** — low LSTM MSE (zero velocity is self-consistent); primary detector is kinematic too_slow flag

---

## Combining with VAE and kinematic flags

```python
flag_abnormal = (
    vae_recon_mse   > vae_p95_threshold    # C2: whole-window plausibility
    or lstm_pred_mse > lstm_p95_threshold  # C3: step-to-step plausibility
    or any_kinematic_flag                  # hard physical rules
)
```

The LSTM adds value over the VAE for corruptions that preserve global statistics but violate local transitions (e.g. sudden turns). The VAE adds value for corruptions that are globally implausible but locally smooth (e.g. random walk). Kinematic flags provide hard detection for position jumps and stationarity that neither learned model catches reliably.
