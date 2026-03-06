"""Feature engineering for linear function approximation.

Contains:
- compute_features_batch_jax: standard 54-dim hand-crafted features
- compute_features_extended: extended 65-dim features with strategic additions
- make_tile_coding_features: tile-coded features (Sutton Ch 9.5.3)
- NumPy versions for tournament agent
"""

import jax
import jax.numpy as jnp


N_FEATURES = 54
N_FEATURES_EXTENDED = 65


def make_tile_coding_features(n_tilings=4, n_tiles=8):
    """Build a tile-coded feature function with overlapping offset tilings. """
    offsets = [k / (n_tilings * n_tiles) for k in range(n_tilings)]
    n_tiles_sq = n_tiles * n_tiles
    n_features = n_tilings * 3 * n_tiles_sq

    def _single(v, t, b1, b2, bid):
        safe_b = jnp.where(b1 > 1e-10, b1, 1.0)
        bf = bid / safe_b
        v_n = (v - 10.) / 10.
        b_n = b1 / jnp.maximum(b1 + b2, 1e-8)
        t_n = t / 9.

        parts = []
        for off in offsets:
            def _bin(x):
                return jnp.floor((x + off) * n_tiles).astype(jnp.int32) % n_tiles

            tx = _bin(bf)
            parts.append(jax.nn.one_hot(tx * n_tiles + _bin(v_n), n_tiles_sq))
            parts.append(jax.nn.one_hot(tx * n_tiles + _bin(b_n), n_tiles_sq))
            parts.append(jax.nn.one_hot(_bin(t_n) * n_tiles + _bin(b_n), n_tiles_sq))

        return jnp.concatenate(parts)

    def feature_fn(v, t, b1, b2, candidates):
        return jax.vmap(lambda bid: _single(v, t, b1, b2, bid))(candidates)

    return feature_fn, n_features


def _features_single(v, t, own_b, opp_b, bid):
    """Compute 54-dim feature vector for a single (state, action) pair."""
    v = jnp.asarray(v, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)
    own_b = jnp.asarray(own_b, dtype=jnp.float32)
    opp_b = jnp.asarray(opp_b, dtype=jnp.float32)

    # State features (36-dim)
    v_idx = jnp.clip(jnp.floor(v - 10.0).astype(jnp.int32), 0, 9)
    t_idx = jnp.clip(t // 2, 0, 4)
    ob_idx = jnp.clip(jnp.floor(own_b / 10.0).astype(jnp.int32), 0, 9)
    op_idx = jnp.clip(jnp.floor(opp_b / 10.0).astype(jnp.int32), 0, 9)
    bdiff = own_b - opp_b

    state = jnp.concatenate([
        jax.nn.one_hot(v_idx, 10),
        jax.nn.one_hot(t_idx, 5),
        jax.nn.one_hot(ob_idx, 10),
        jax.nn.one_hot(op_idx, 10),
        jnp.array([bdiff]),
    ])

    # Action features (11-dim)
    safe_b = jnp.where(own_b > 1e-10, own_b, 1.0)
    a_norm = bid / safe_b
    a_idx = jnp.where(a_norm >= 1.0, 10,
                      jnp.clip(jnp.floor(a_norm * 10).astype(jnp.int32), 0, 9))
    action = jax.nn.one_hot(a_idx, 11)

    # Interaction features (7-dim)
    low_v = (v <= 13.0).astype(jnp.float32)
    mid_v = ((v > 13.0) & (v < 17.0)).astype(jnp.float32)
    hi_v = (v >= 17.0).astype(jnp.float32)
    early = (t <= 3).astype(jnp.float32)
    mid_t = ((t >= 4) & (t <= 7)).astype(jnp.float32)
    late = (t >= 8).astype(jnp.float32)

    ixn = jnp.array([
        a_norm * low_v, a_norm * mid_v, a_norm * hi_v,
        a_norm * early, a_norm * mid_t, a_norm * late,
        a_norm * bdiff,
    ])

    return jnp.concatenate([state, action, ixn])


def compute_features_batch_jax(v, t, own_b, opp_b, candidates):
    """Compute 54-dim feature vectors for a batch of candidate actions. """
    return jax.vmap(lambda bid: _features_single(v, t, own_b, opp_b, bid))(candidates)


def _features_extended_single(v, t, own_b, opp_b, bid):
    """Compute 65-dim feature vector for a single (state, action) pair."""
    base = _features_single(v, t, own_b, opp_b, bid)

    v = jnp.asarray(v, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)
    own_b = jnp.asarray(own_b, dtype=jnp.float32)
    opp_b = jnp.asarray(opp_b, dtype=jnp.float32)

    safe_b = jnp.where(own_b > 1e-10, own_b, 1.0)
    a_norm = bid / safe_b
    budget_ratio = own_b / jnp.maximum(own_b + opp_b, 1e-8)
    rounds_left = jnp.maximum(10.0 - t, 1.0)
    v_norm = (v - 10.0) / 10.0
    t_norm = t / 9.0
    is_last = (t == 9).astype(jnp.float32)
    budget_adv = (own_b - opp_b) / 100.0

    extra = jnp.array([
        budget_ratio,
        own_b / (rounds_left * 100.0),
        opp_b / (rounds_left * 100.0),
        is_last,
        bid / (jnp.maximum(v, 1e-8) * 10.0),
        a_norm * budget_ratio,
        a_norm * t_norm,
        a_norm * is_last,
        v_norm,
        v_norm * budget_adv,
        t_norm,
    ])

    return jnp.concatenate([base, extra])


def compute_features_extended(v, t, own_b, opp_b, candidates):
    """Extended 65-dim feature vectors: standard 54 + 11 additions."""
    return jax.vmap(lambda bid: _features_extended_single(v, t, own_b, opp_b, bid))(candidates)


# ── NumPy versions for tournament agent ──

def compute_features_numpy(v, t, own_b, opp_b, candidates):
    """NumPy-only version of the 54-dim features for tournament agent."""
    import numpy as np

    n = len(candidates)

    v_idx = min(max(int(v - 10.0), 0), 9)
    t_idx = min(t // 2, 4)
    ob_idx = min(max(int(own_b / 10.0), 0), 9)
    op_idx = min(max(int(opp_b / 10.0), 0), 9)
    bdiff = own_b - opp_b

    state_feats = np.zeros(36)
    state_feats[v_idx] = 1.0
    state_feats[10 + t_idx] = 1.0
    state_feats[15 + ob_idx] = 1.0
    state_feats[25 + op_idx] = 1.0
    state_feats[35] = bdiff

    safe_b = own_b if own_b > 1e-10 else 1.0
    a_norms = candidates / safe_b
    a_idxs = np.where(a_norms >= 1.0, 10,
                      np.clip((a_norms * 10).astype(int), 0, 9))
    action_bins = np.eye(11)[a_idxs]

    low_v = float(v <= 13.0)
    mid_v = float(13.0 < v < 17.0)
    hi_v = float(v >= 17.0)
    early_t = float(t <= 3)
    mid_t = float(4 <= t <= 7)
    late_t = float(t >= 8)

    ixn = np.column_stack([
        a_norms * low_v, a_norms * mid_v, a_norms * hi_v,
        a_norms * early_t, a_norms * mid_t, a_norms * late_t,
        a_norms * bdiff,
    ])

    state_bc = np.tile(state_feats, (n, 1))
    return np.hstack([state_bc, action_bins, ixn])


def compute_features_extended_numpy(v, t, own_b, opp_b, candidates):
    """NumPy-only version of the 65-dim extended features for tournament agent."""
    import numpy as np

    n = len(candidates)
    base = compute_features_numpy(v, t, own_b, opp_b, candidates)

    safe_b = own_b if own_b > 1e-10 else 1.0
    a_norms = candidates / safe_b
    budget_ratio = own_b / max(own_b + opp_b, 1e-8)
    rounds_left = max(10.0 - t, 1.0)
    v_norm = (v - 10.0) / 10.0
    t_norm = t / 9.0
    is_last = float(t == 9)
    budget_adv = (own_b - opp_b) / 100.0

    extra = np.column_stack([
        np.full(n, budget_ratio),
        np.full(n, own_b / (rounds_left * 100.0)),
        np.full(n, opp_b / (rounds_left * 100.0)),
        np.full(n, is_last),
        candidates / (max(v, 1e-8) * 10.0),
        a_norms * budget_ratio,
        a_norms * t_norm,
        a_norms * is_last,
        np.full(n, v_norm),
        np.full(n, v_norm * budget_adv),
        np.full(n, t_norm),
    ])

    return np.hstack([base, extra])
