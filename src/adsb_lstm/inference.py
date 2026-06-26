from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import SequenceDataset
from .model import build_model
from .utils import get_device


def load_checkpoint(
    path: str | Path,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict]:
    if device is None:
        device = get_device()
    ckpt   = torch.load(path, map_location=device)
    config = ckpt["config"]
    model  = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def compute_prediction_errors(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    records: list[dict] = []
    seq_idx = 0
    with torch.no_grad():
        for x_input, y_target in loader:
            x_input  = x_input.to(device)
            y_target = y_target.to(device)
            y_pred   = model(x_input)                       # (B, 29, 4)
            err      = (y_pred - y_target) ** 2             # (B, 29, 4)
            total_mse     = err.mean(dim=(1, 2))            # (B,)
            pos_mse       = err[:, :, :2].mean(dim=(1, 2))  # E, N
            vel_mse       = err[:, :, 2:].mean(dim=(1, 2))  # vE, vN
            final_step_mse = err[:, -1, :].mean(dim=1)      # last predicted step
            for i in range(len(x_input)):
                records.append({
                    "sequence_index":  seq_idx + i,
                    "total_mse":       total_mse[i].item(),
                    "pos_mse":         pos_mse[i].item(),
                    "vel_mse":         vel_mse[i].item(),
                    "final_step_mse":  final_step_mse[i].item(),
                })
            seq_idx += len(x_input)
    return pd.DataFrame(records)


def score_sequences(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 1024,
    device: torch.device | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    if device is None:
        device = get_device()
    ds     = SequenceDataset(X, max_samples=max_samples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return compute_prediction_errors(model, loader, device)
