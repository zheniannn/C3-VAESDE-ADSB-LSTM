from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import load_config, set_seed, ensure_dir
from .dataset import SequenceDataset
from .model import build_model
from .utils import get_device, count_parameters, describe_array


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> float:
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    n_seq = 0
    for x_input, y_target in loader:
        x_input = x_input.to(device)
        y_target = y_target.to(device)
        optimizer.zero_grad()
        y_pred = model(x_input)
        loss = criterion(y_pred, y_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += loss.item() * len(x_input)
        n_seq += len(x_input)
    return total_loss / n_seq if n_seq > 0 else 0.0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    n_seq = 0
    with torch.no_grad():
        for x_input, y_target in loader:
            x_input = x_input.to(device)
            y_target = y_target.to(device)
            y_pred = model(x_input)
            loss = criterion(y_pred, y_target)
            total_loss += loss.item() * len(x_input)
            n_seq += len(x_input)
    return total_loss / n_seq if n_seq > 0 else 0.0


def fit_model(config: dict) -> None:
    set_seed(config["seed"])
    out_dir = ensure_dir(config["output_dir"])
    device = get_device()
    print(f"Device: {device}", flush=True)

    data_dir = Path(config["data_dir"])
    train_path = data_dir / config["train_file"]
    test_path  = data_dir / config["test_file"]

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing X_train.npy or X_test.npy. "
            "Copy them from C1-VAESDE-ADSB-PREPROCESSING into data/."
        )

    print("Loading data ...", flush=True)
    X_train_mmap = np.load(train_path, mmap_mode="r")
    X_test_mmap  = np.load(test_path,  mmap_mode="r")
    describe_array("X_train", X_train_mmap)
    describe_array("X_test",  X_test_mmap)

    debug      = config.get("debug_mode", False)
    train_max  = config.get("debug_train_size") if debug else None
    test_max   = config.get("debug_test_size")  if debug else None

    if debug:
        print(f"Debug mode: {train_max:,} train / {test_max:,} test sequences")
        # SequenceDataset copies the sampled subset into memory
        train_ds = SequenceDataset(X_train_mmap, max_samples=train_max)
        test_ds  = SequenceDataset(X_test_mmap,  max_samples=test_max)
    else:
        # Copy full arrays to float32 in RAM upfront — avoids page-fault storms
        # from random memmap access during shuffled DataLoader iteration.
        print(f"Full mode: {len(X_train_mmap):,} train / {len(X_test_mmap):,} test — copying to RAM ...", flush=True)
        X_train = X_train_mmap[:].astype(np.float32)
        print(f"  X_train in RAM ({X_train.nbytes / 1e9:.2f} GB)", flush=True)
        X_test  = X_test_mmap[:].astype(np.float32)
        print(f"  X_test  in RAM ({X_test.nbytes / 1e9:.2f} GB)", flush=True)
        train_ds = SequenceDataset(X_train)
        test_ds  = SequenceDataset(X_test)

    batch_size  = config.get("batch_size", 1024)
    num_workers = config.get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model     = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 1e-6),
    )
    print(f"Model: {config.get('model_type', 'lstm').upper()}  params={count_parameters(model):,}", flush=True)

    epochs        = config.get("epochs", 20)
    gradient_clip = config.get("gradient_clip", 1.0)

    print(f"\n{'Epoch':>5}  {'Train Loss':>12}  {'Test Loss':>12}", flush=True)
    print("-" * 35, flush=True)

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, gradient_clip)
        te_loss = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "test_loss": te_loss})
        print(f"{epoch:>5}  {tr_loss:>12.6f}  {te_loss:>12.6f}", flush=True)

    ckpt_path = out_dir / "motion_lstm.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config":           config,
        "final_train_loss": history[-1]["train_loss"],
        "final_test_loss":  history[-1]["test_loss"],
        "history":          history,
    }, ckpt_path)
    print(f"\nCheckpoint: {ckpt_path}", flush=True)

    hist_path = out_dir / "history.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"History:    {hist_path}", flush=True)
