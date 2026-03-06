"""Tournament agent — Group B.

Mixture-of-Betas policy trained via NFSP (Neural Fictitious Self-Play)
with diverse asymmetric opponents. Standalone inference using NumPy only.

Architecture:
    16 strategic features -> LayerNorm -> Dense(128,tanh) -> Dense(64,tanh)
    -> Dense(32,tanh) -> Dense(32,tanh) -> Dense(9) -> 3 Beta components
    with mixture weights. Deterministic deployment via weighted mean.
"""
import numpy as np
import os

_dir = os.path.dirname(os.path.abspath(__file__))
_w = np.load(os.path.join(_dir, 'weights.npz'))
_N_MIX = 3


def _features(v, t, B1, B2):
    """16 strategic features from raw state (v, t, B_own, B_opp)."""
    eps = 1e-8
    T = 10.0
    rl = max(T - t, 1.0)
    return np.array([
        v / 20.0, t / 9.0, B1 / 100.0, B2 / 100.0,
        (B1 - B2) / 100.0,
        min(B1 / (B2 + eps), 5.0),
        min(v / (B1 + eps), 2.0),
        min(v / (B2 + eps), 2.0),
        B1 / rl / 15.0, B2 / rl / 15.0,
        (100.0 - B1) / (t + 1) / 15.0,
        (100.0 - B2) / (t + 1) / 15.0,
        v * rl / (B1 + B2 + eps),
        float(t >= 9),
        max(min((B1 / rl - B2 / rl) / 15.0, 2.0), -2.0),
        rl / 10.0,
    ], dtype=np.float64)


def _forward(x):
    """Forward pass returning mixture-of-Betas parameters."""
    m, s = x.mean(), x.std()
    x = (x - m) / (s + 1e-5) * _w['ln_scale'] + _w['ln_bias']
    for i in range(3):
        x = np.tanh(x @ _w[f'dense_{i}_W'] + _w[f'dense_{i}_b'])
    h = np.tanh(x @ _w['actor_h_W'] + _w['actor_h_b'])
    raw = h @ _w['actor_out_W'] + _w['actor_out_b']
    raw = raw.reshape(_N_MIX, 3)
    alphas = np.log1p(np.exp(raw[:, 0])) + 1.0
    betas = np.log1p(np.exp(raw[:, 1])) + 1.0
    logits = raw[:, 2]
    logits = logits - logits.max()
    w = np.exp(logits)
    w = w / w.sum()
    return alphas, betas, w


def policyB(t, v, B_own, B_opp):
    """Tournament policy for Group B.

    Args:
        t:     time index (0-9)
        v:     prize value (~U[10,20])
        B_own: own remaining budget
        B_opp: opponent remaining budget

    Returns:
        bid: a feasible bid in [0, B_own]
    """
    if B_own <= 0:
        return 0.0
    if t >= 9:
        return float(B_own)

    x = _features(v, float(t), float(B_own), float(B_opp))
    alphas, betas, weights = _forward(x)

    means = alphas / (alphas + betas)
    frac = float(np.sum(weights * means))
    frac = np.clip(frac + np.random.normal(0, 0.02), 0.0, 1.0)

    bid = frac * B_own
    return float(np.clip(bid, 0, B_own))
