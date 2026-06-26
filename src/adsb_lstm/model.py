from __future__ import annotations

import torch
import torch.nn as nn


class MotionLSTM(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 29, 4)  →  out: (B, 29, 4)
        out, _ = self.lstm(x)
        return self.head(out)


class MotionGRU(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out)


def build_model(config: dict) -> nn.Module:
    model_type = config.get("model_type", "lstm")
    kwargs = dict(
        input_dim=config.get("n_features", 4),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.1),
    )
    if model_type == "lstm":
        return MotionLSTM(**kwargs)
    if model_type == "gru":
        return MotionGRU(**kwargs)
    raise ValueError(f"Unknown model_type: {model_type!r}. Choose 'lstm' or 'gru'.")
