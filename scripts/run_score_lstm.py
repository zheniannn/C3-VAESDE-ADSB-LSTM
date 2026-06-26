"""Score train/test sequences and compute detection thresholds.

Usage:
    python scripts/run_score_lstm.py
    python scripts/run_score_lstm.py --config configs/lstm_default.yaml --checkpoint outputs/lstm/motion_lstm.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adsb_lstm.config import load_config
from adsb_lstm.inference import load_checkpoint, score_sequences
from adsb_lstm.reporting import save_dataframe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score sequences and compute thresholds")
    p.add_argument("--config",     default="configs/lstm_default.yaml")
    p.add_argument("--checkpoint", default="outputs/lstm/motion_lstm.pt")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    config   = load_config(args.config)
    out_dir  = Path(config["output_dir"])
    data_dir = Path(config["data_dir"])

    model, _ = load_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    X_train = np.load(data_dir / config["train_file"], mmap_mode="r")
    X_test  = np.load(data_dir / config["test_file"],  mmap_mode="r")

    batch_size = config.get("batch_size", 1024)

    print("Scoring train sequences ...")
    train_errors = score_sequences(model, X_train, batch_size=batch_size)
    save_dataframe(train_errors, out_dir / "train_prediction_errors.csv")
    print(f"  Saved {len(train_errors):,} rows")

    print("Scoring test sequences ...")
    test_errors = score_sequences(model, X_test, batch_size=batch_size)
    save_dataframe(test_errors, out_dir / "test_prediction_errors.csv")
    print(f"  Saved {len(test_errors):,} rows")

    quantiles = config.get("threshold_quantiles", [0.90, 0.95, 0.99])
    train_mse = train_errors["total_mse"].values
    test_mse  = test_errors["total_mse"].values

    print(f"\n{'Quantile':>10}  {'Threshold':>12}  {'Test flag rate':>15}")
    print("-" * 42)
    rows = []
    for q in quantiles:
        thr       = float(np.quantile(train_mse, q))
        flag_rate = float((test_mse > thr).mean())
        label     = f"p{int(q * 100)}"
        rows.append({"quantile": label, "threshold": thr, "test_flag_rate": flag_rate})
        print(f"{label:>10}  {thr:>12.6f}  {flag_rate:>14.1%}")

    thr_path = out_dir / "lstm_thresholds.csv"
    save_dataframe(pd.DataFrame(rows), thr_path)
    print(f"\nThresholds saved: {thr_path}")

    print(f"\nTrain MSE  mean={train_mse.mean():.6f}  p50={np.median(train_mse):.6f}  p95={np.quantile(train_mse, 0.95):.6f}")
    print(f"Test  MSE  mean={test_mse.mean():.6f}  p50={np.median(test_mse):.6f}  p95={np.quantile(test_mse, 0.95):.6f}")


if __name__ == "__main__":
    main()
