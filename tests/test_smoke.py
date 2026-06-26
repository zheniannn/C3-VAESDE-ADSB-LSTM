"""Smoke tests — no real data required. Uses synthetic (128, 30, 4) arrays."""
import numpy as np
import pytest
import torch


def make_synthetic(n: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 30, 4)).astype(np.float32)


# ── Dataset ────────────────────────────────────────────────────────────────────

def test_dataset_shapes():
    from adsb_lstm.dataset import SequenceDataset
    ds = SequenceDataset(make_synthetic())
    x_input, y_target = ds[0]
    assert x_input.shape  == (29, 4)
    assert y_target.shape == (29, 4)
    assert x_input.dtype  == torch.float32


def test_dataset_len():
    from adsb_lstm.dataset import SequenceDataset
    assert len(SequenceDataset(make_synthetic(64))) == 64


def test_dataset_max_samples():
    from adsb_lstm.dataset import SequenceDataset
    ds = SequenceDataset(make_synthetic(128), max_samples=32)
    assert len(ds) == 32


def test_dataset_bad_shape():
    from adsb_lstm.dataset import SequenceDataset
    with pytest.raises(ValueError):
        SequenceDataset(np.zeros((10, 20, 4)))


# ── Model ──────────────────────────────────────────────────────────────────────

def test_motion_lstm_forward():
    from adsb_lstm.model import MotionLSTM
    model = MotionLSTM(input_dim=4, hidden_dim=32, num_layers=1, dropout=0.0)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(8, 29, 4))
    assert out.shape == (8, 29, 4)


def test_motion_gru_forward():
    from adsb_lstm.model import MotionGRU
    model = MotionGRU(input_dim=4, hidden_dim=32, num_layers=1, dropout=0.0)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(8, 29, 4))
    assert out.shape == (8, 29, 4)


def test_build_model_lstm_and_gru():
    from adsb_lstm.model import build_model
    for mt in ("lstm", "gru"):
        cfg = {"model_type": mt, "n_features": 4, "hidden_dim": 32, "num_layers": 1, "dropout": 0.0}
        assert build_model(cfg) is not None


def test_build_model_unknown_raises():
    from adsb_lstm.model import build_model
    with pytest.raises(ValueError):
        build_model({"model_type": "transformer"})


# ── Training / evaluation loop ─────────────────────────────────────────────────

def test_train_eval_loop():
    from adsb_lstm.dataset import SequenceDataset
    from adsb_lstm.model import MotionLSTM
    from adsb_lstm.training import train_one_epoch, evaluate
    from torch.utils.data import DataLoader

    device = torch.device("cpu")
    ds     = SequenceDataset(make_synthetic(64))
    loader = DataLoader(ds, batch_size=16)
    model  = MotionLSTM(input_dim=4, hidden_dim=16, num_layers=1, dropout=0.0).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

    tr_loss = train_one_epoch(model, loader, opt, device, gradient_clip=1.0)
    te_loss = evaluate(model, loader, device)
    assert tr_loss > 0
    assert te_loss > 0


# ── Inference ──────────────────────────────────────────────────────────────────

def test_compute_prediction_errors_columns():
    from adsb_lstm.model import MotionLSTM
    from adsb_lstm.inference import score_sequences

    device = torch.device("cpu")
    model  = MotionLSTM(input_dim=4, hidden_dim=16, num_layers=1, dropout=0.0).to(device)
    model.eval()
    df = score_sequences(model, make_synthetic(32), batch_size=16, device=device)
    assert set(df.columns) >= {"sequence_index", "total_mse", "pos_mse", "vel_mse", "final_step_mse"}
    assert len(df) == 32
    assert (df["total_mse"] >= 0).all()


# ── Corruptions ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fn", [
    lambda x: __import__("adsb_lstm.corruption", fromlist=["speed_scale"]).speed_scale(x, 1.5),
    lambda x: __import__("adsb_lstm.corruption", fromlist=["speed_scale"]).speed_scale(x, 2.0),
    __import__("adsb_lstm.corruption", fromlist=["random_walk_velocity"]).random_walk_velocity,
    __import__("adsb_lstm.corruption", fromlist=["sudden_turn_90"]).sudden_turn_90,
    __import__("adsb_lstm.corruption", fromlist=["position_jump"]).position_jump,
    __import__("adsb_lstm.corruption", fromlist=["stationary_clutter"]).stationary_clutter,
])
def test_corruption_shape_and_finite(fn):
    x   = make_synthetic(16)
    out = fn(x)
    assert out.shape == x.shape
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
