"""Episode generation and loss functions.

Contains:
- generate_episodes: JAX-vectorized episode generation
- mse: mean squared error loss
- huber: Huber loss (smooth L1)
- Shared constants: N_ACTIONS, T, CLIP
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from sections.features import compute_features_batch_jax
from sections.policies import default_player_2, uniform_policy


# Shared constants used across all agents.
N_ACTIONS = 21
T = 10
CLIP = 100  # default gradient clip; tune via clip= kwarg in MC/TD/SARSA


# Loss functions shared by MC, TD, and SARSA training routines.

def mse(theta, phi, target):
    """Mean squared error loss."""
    pred = jnp.dot(phi, theta)
    return 0.5 * (target - pred) ** 2


def huber(theta, phi, target, delta=1.0):
    """Huber loss (smooth L1)."""
    pred = jnp.dot(phi, theta)
    d = target - pred
    abs_d = jnp.abs(d)
    return jnp.where(abs_d <= delta, 0.5 * d ** 2, delta * (abs_d - 0.5 * delta))




@partial(jit, static_argnames=(
    'n_episodes', 'opponent_fn', 'prediction', 'n_actions', 'feature_fn', 'zero_sum'
))
def generate_episodes(n_episodes, key, epsilon, theta,
                      opponent_fn=uniform_policy, prediction=True,
                      n_actions=N_ACTIONS, feature_fn=compute_features_batch_jax,
                      zero_sum=False):
    """Generate a batch of episodes using JAX vectorization.  """

    def one_episode(key):
        def step(carry, t):
            key, b1, b2 = carry
            key, kv, kn1, kn2, ke = jax.random.split(key, 5)

            v = jax.random.uniform(kv, minval=10.0, maxval=20.0)
            cands = jnp.linspace(0.0, b1, n_actions)
            phis = feature_fn(v, t, b1, b2, cands)

            if prediction:
                # Both players follow fixed strategies for offline data.
                bid1 = default_player_2(t, v, b1, b2, kn1)
                idx = jnp.argmin(jnp.abs(cands - bid1))
                bid2 = opponent_fn(t, v, b2, b1, kn2)
            else:
                # Player 1 selects actions using epsilon-greedy over Q-values.
                q_vals = phis @ theta

                greedy = jnp.argmax(q_vals)
                ke1, ke2 = jax.random.split(ke)
                rand = jax.random.randint(ke1, (), 0, n_actions)
                idx = jnp.where(jax.random.uniform(ke2) < epsilon, rand, greedy)
                bid1 = cands[idx]
                bid2 = default_player_2(t, v, b1, b2, kn2)

            phi1 = phis[idx]
            base_reward = jnp.where(bid1 > bid2, v,
                          jnp.where(bid1 == bid2, v * 0.5, 0.0))
            reward = base_reward - (v * 0.5 if zero_sum else 0.0)

            return (key, b1 - bid1, b2 - bid2), (phi1, reward, jnp.array([b1, b2]), bid1, bid2)

        _, (phis, rewards, budgets, bids1, bids2) = jax.lax.scan(
            step,
            (key, jnp.float32(100.0), jnp.float32(100.0)),
            jnp.arange(10)
        )
        return phis, rewards, budgets, bids1, bids2

    keys = jax.random.split(key, n_episodes)
    phis, rewards, budgets, bids1, bids2 = jax.vmap(one_episode)(keys)
    return phis, rewards, budgets, bids1, bids2
