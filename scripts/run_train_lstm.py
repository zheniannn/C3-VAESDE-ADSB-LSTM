"""Train MotionLSTM on ADS-B ENU trajectory windows.

Usage:
    python scripts/run_train_lstm.py
    python scripts/run_train_lstm.py --config configs/lstm_default.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adsb_lstm.config import load_config
from adsb_lstm.training import fit_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MotionLSTM")
    p.add_argument("--config", default="configs/lstm_default.yaml", help="Path to YAML config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    fit_model(config)


if __name__ == "__main__":
    main()
