"""Stress-test the trained MotionLSTM against corruption types.

Usage:
    python scripts/run_stress_test_lstm.py
    python scripts/run_stress_test_lstm.py --checkpoint outputs/lstm/motion_lstm.pt --max-samples 50000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adsb_lstm.config import load_config
from adsb_lstm.inference import load_checkpoint, score_sequences
from adsb_lstm.corruption import (
    speed_scale, random_walk_velocity, sudden_turn_90, position_jump, stationary_clutter,
)
from adsb_lstm.reporting import save_dataframe
from adsb_lstm.utils import get_device

CORRUPTIONS = [
    ("clean",                lambda x: x),
    ("speed_scaled_1.5",     lambda x: speed_scale(x, factor=1.5)),
    ("speed_scaled_2.0",     lambda x: speed_scale(x, factor=2.0)),
    ("random_walk_velocity", random_walk_velocity),
    ("sudden_turn_90",       sudden_turn_90),
    ("position_jump",        position_jump),
    ("stationary_clutter",   stationary_clutter),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stress-test MotionLSTM")
    p.add_argument("--config",      default="configs/lstm_default.yaml")
    p.add_argument("--checkpoint",  default="outputs/lstm/motion_lstm.pt")
    p.add_argument("--max-samples", type=int, default=50_000)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    out_dir = Path(config["output_dir"])

    model, _ = load_checkpoint(args.checkpoint)
    device   = get_device()
    print(f"Loaded checkpoint: {args.checkpoint}")

    thr_path = out_dir / "lstm_thresholds.csv"
    if not thr_path.exists():
        raise FileNotFoundError(
            f"Thresholds not found at {thr_path}. Run run_score_lstm.py first."
        )
    thr_df    = pd.read_csv(thr_path)
    p95_row   = thr_df[thr_df["quantile"] == "p95"]
    if p95_row.empty:
        raise ValueError("p95 row not found in lstm_thresholds.csv")
    p95_thr = float(p95_row["threshold"].iloc[0])
    print(f"Train p95 threshold: {p95_thr:.6f}\n")

    data_dir    = Path(config["data_dir"])
    X_test_full = np.load(data_dir / config["test_file"], mmap_mode="r")
    n           = min(args.max_samples, len(X_test_full))
    rng         = np.random.default_rng(config.get("seed", 42))
    idx         = np.sort(rng.choice(len(X_test_full), n, replace=False))
    X_test      = X_test_full[idx].copy()
    print(f"Using {n:,} test sequences")

    batch_size = config.get("batch_size", 1024)

    print(f"\n  {'Case':<25}  {'Mean MSE':>10}  {'p95 MSE':>10}  {'Det%':>8}")
    print("  " + "-" * 60)

    rows = []
    for name, corrupt_fn in CORRUPTIONS:
        X_c    = corrupt_fn(X_test)
        errors = score_sequences(model, X_c, batch_size=batch_size, device=device)
        mse    = errors["total_mse"].values
        det    = float((mse > p95_thr).mean())
        rows.append({
            "case":                        name,
            "mean_total_mse":              float(mse.mean()),
            "p95_total_mse":               float(np.quantile(mse, 0.95)),
            "detection_rate_at_train_p95": det,
        })
        print(f"  {name:<25}  {mse.mean():>10.6f}  {float(np.quantile(mse, 0.95)):>10.6f}  {det:>7.1%}")

    summary = pd.DataFrame(rows)
    save_dataframe(summary, out_dir / "lstm_stress_summary.csv")
    print(f"\nSaved: {out_dir / 'lstm_stress_summary.csv'}")


if __name__ == "__main__":
    main()
