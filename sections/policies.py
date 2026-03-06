"""Opponent strategies (JAX and NumPy versions).

JAX versions accept a PRNG key for reproducible stochastic policies.
NumPy versions use global numpy random state for the tournament agent.
"""

import jax
import jax.numpy as jnp
import numpy as np


def uniform_policy(t, v, own_budget, opp_budget, key):
    """Uniform random bid in [0, own_budget]."""
    return jax.random.uniform(key, minval=0.0, maxval=own_budget)


def default_player_2(t, v, own_budget, opp_budget, key):
    """Default opponent: value-aware sigmoid bidding with noise."""
    T = 10
    noise = jax.random.normal(key) * 3.0
    noise = jnp.where(t < T - 1, noise, 0.0)
    numerator = jnp.exp(v - 15.0)
    denominator = numerator + (T - t - 1)
    bid = (numerator / denominator) * own_budget + noise
    return jnp.clip(bid, 0.0, own_budget)


# NumPy versions for the tournament agent and the class-based environment.

def uniform_policy_numpy(t, v, own_budget, opp_budget):
    """Uniform random bid using numpy."""
    return np.random.uniform(0, own_budget)


def default_player_2_numpy(t, v, own_budget, opp_budget):
    """Default opponent strategy using numpy."""
    T = 10
    noise = np.random.normal(0, 3) if t < T - 1 else 0.0
    numerator = np.exp(v - 15.0)
    denominator = numerator + (T - t - 1)
    bid = (numerator / denominator) * own_budget + noise
    return np.clip(bid, 0, own_budget)
