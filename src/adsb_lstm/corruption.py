from __future__ import annotations

import numpy as np

# All functions accept (N, 30, 4) normalised arrays [E, N, vE, vN] and return the same shape.
#
# Positions and velocities are normalised to zero mean / unit std on the training split.
# _DT_NORM approximates the normalised time step: std_v * dt_s / std_pos ≈ 50*10/20000 = 0.025.
# Used only when recomputing positions from velocities — it keeps positions in a plausible range.

_DT_NORM = 0.025


def speed_scale(x: np.ndarray, factor: float = 1.5) -> np.ndarray:
    out = x.copy()
    out[:, :, 2] *= factor   # scale vE
    out[:, :, 3] *= factor   # scale vN
    # Recompute E, N using scaled velocity so position/velocity are consistent
    out[:, 1:, 0] = out[:, 0:1, 0] + np.cumsum(out[:, :-1, 2] * _DT_NORM, axis=1)
    out[:, 1:, 1] = out[:, 0:1, 1] + np.cumsum(out[:, :-1, 3] * _DT_NORM, axis=1)
    return out


def random_walk_velocity(x: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = x.copy()
    out[:, :, 2] = rng.standard_normal((out.shape[0], out.shape[1]))  # vE ← noise
    out[:, :, 3] = rng.standard_normal((out.shape[0], out.shape[1]))  # vN ← noise
    out[:, 1:, 0] = out[:, 0:1, 0] + np.cumsum(out[:, :-1, 2] * _DT_NORM, axis=1)
    out[:, 1:, 1] = out[:, 0:1, 1] + np.cumsum(out[:, :-1, 3] * _DT_NORM, axis=1)
    return out


def sudden_turn_90(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    mid = out.shape[1] // 2
    vE = out[:, mid:, 2].copy()
    vN = out[:, mid:, 3].copy()
    out[:, mid:, 2] = -vN   # 90-degree rotation: vE_new = -vN
    out[:, mid:, 3] =  vE   #                     vN_new =  vE
    # Recompute positions from mid onward
    if mid + 1 < out.shape[1]:
        out[:, mid + 1:, 0] = out[:, mid:mid + 1, 0] + np.cumsum(out[:, mid:-1, 2] * _DT_NORM, axis=1)
        out[:, mid + 1:, 1] = out[:, mid:mid + 1, 1] + np.cumsum(out[:, mid:-1, 3] * _DT_NORM, axis=1)
    return out


def position_jump(x: np.ndarray, jump_normalised: float = 2.0) -> np.ndarray:
    out = x.copy()
    mid = out.shape[1] // 2
    out[:, mid:, 0] += jump_normalised   # shift E from midpoint
    out[:, mid:, 1] += jump_normalised   # shift N from midpoint
    return out


def stationary_clutter(x: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(42)
    out = x.copy()
    tiny = 0.01
    out[:, :, 0] = out[:, 0:1, 0] + rng.normal(0, tiny, (out.shape[0], out.shape[1]))
    out[:, :, 1] = out[:, 0:1, 1] + rng.normal(0, tiny, (out.shape[0], out.shape[1]))
    out[:, :, 2] = rng.normal(0, tiny, (out.shape[0], out.shape[1]))
    out[:, :, 3] = rng.normal(0, tiny, (out.shape[0], out.shape[1]))
    return out
