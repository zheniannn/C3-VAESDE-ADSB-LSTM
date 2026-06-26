from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Wraps a (N, 30, 4) array; returns (x_input, y_target) pairs for next-step prediction."""

    def __init__(self, data: np.ndarray, max_samples: int | None = None):
        if data.ndim != 3 or data.shape[1:] != (30, 4):
            raise ValueError(f"Expected shape (N, 30, 4), got {data.shape}")
        if max_samples is not None and max_samples < len(data):
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(len(data), max_samples, replace=False))
            # Copy selected rows — needed when data is a memmap
            self._data = data[idx].copy()
        else:
            self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self._data[idx].astype(np.float32)
        x_input = torch.from_numpy(seq[:-1, :])   # (29, 4)
        y_target = torch.from_numpy(seq[1:, :])    # (29, 4)
        return x_input, y_target
