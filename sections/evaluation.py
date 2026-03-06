"""Evaluation utilities.

Functions for evaluating trained agents:
- evaluate_theta: greedy evaluation of a trained theta against default_player_2
- evaluate_prediction: prediction evaluation (predicted vs actual returns)
- run_evaluation: batch evaluation over checkpointed thetas
- run_prediction_evaluation: batch prediction evaluation
- run_winrate_evaluation: win rate evaluation
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from sections.features import compute_features_batch_jax, N_FEATURES
from sections.policies import default_player_2, uniform_policy
from sections.episodes import N_ACTIONS

V_MIN = 10
V_MAX = 20


# Inner episode functions are plain (non-JIT) for composition with jax.vmap.

def _eval_episode(theta, key,
                  n_actions=N_ACTIONS, feature_fn=compute_features_batch_jax):
    """One greedy evaluation episode for a single theta."""
    def step(carry, t):
        key, b1, b2 = carry
        key, kv, kn = jax.random.split(key, 3)

        v = jax.random.uniform(kv, minval=V_MIN, maxval=V_MAX)
        cands = jnp.linspace(0.0, b1, n_actions)
        phis = feature_fn(v, t, b1, b2, cands)

        q_vals = phis @ theta

        idx = jnp.where(b1 <= 1e-10, jnp.int32(0), jnp.argmax(q_vals))
        bid1 = cands[idx]

        noise = jnp.where(t < 9, jax.random.normal(kn) * 3.0, 0.0)
        num = jnp.exp(v - 15.0)
        bid2 = jnp.clip((num / (num + (9.0 - t))) * b2 + noise, 0.0, b2)

        reward = jnp.where(bid1 > bid2, v,
                 jnp.where(bid1 == bid2, v * 0.5, 0.0))

        return (key, b1 - bid1, b2 - bid2), reward

    _, rewards = jax.lax.scan(
        step,
        (key, jnp.float32(100.0), jnp.float32(100.0)),
        jnp.arange(10)
    )
    return rewards.sum()


def _pred_episode(theta, key,
                  n_actions=N_ACTIONS, feature_fn=compute_features_batch_jax):
    """One prediction evaluation episode. Returns (total_reward, predicted_value_at_t0)."""
    def step(carry, t):
        key, b1, b2 = carry
        key, kv, kn1, kn2 = jax.random.split(key, 4)

        v = jax.random.uniform(kv, minval=10.0, maxval=20.0)
        bid1 = default_player_2(t, v, b1, b2, kn1)
        bid2 = uniform_policy(t, v, b2, b1, kn2)

        cands = jnp.linspace(0.0, b1, n_actions)
        phis = feature_fn(v, t, b1, b2, cands)
        idx = jnp.argmin(jnp.abs(cands - bid1))

        pred = jnp.dot(theta, phis[idx])

        reward = jnp.where(bid1 > bid2, v,
                           jnp.where(bid1 == bid2, v * 0.5, 0.0))

        return (key, b1 - bid1, b2 - bid2), (reward, pred)

    _, (rewards, preds) = jax.lax.scan(
        step, (key, jnp.float32(100.0), jnp.float32(100.0)), jnp.arange(10)
    )
    return rewards.sum(), preds[0]


def _winrate_episode(theta, key,
                     n_actions=N_ACTIONS, feature_fn=compute_features_batch_jax):
    """One greedy evaluation episode tracking win rate."""
    def step(carry, t):
        key, b1, b2 = carry
        key, kv, kn = jax.random.split(key, 3)

        v = jax.random.uniform(kv, minval=V_MIN, maxval=V_MAX)
        cands = jnp.linspace(0.0, b1, n_actions)
        phis = feature_fn(v, t, b1, b2, cands)

        q_vals = phis @ theta

        idx = jnp.where(b1 <= 1e-10, jnp.int32(0), jnp.argmax(q_vals))
        bid1 = cands[idx]

        noise = jnp.where(t < 9, jax.random.normal(kn) * 3.0, 0.0)
        num = jnp.exp(v - 15.0)
        bid2 = jnp.clip((num / (num + (9.0 - t))) * b2 + noise, 0.0, b2)

        reward = jnp.where(bid1 > bid2, v,
                 jnp.where(bid1 == bid2, v * 0.5, 0.0))
        won = jnp.float32(bid1 > bid2)

        return (key, b1 - bid1, b2 - bid2), (reward, won)

    _, (rewards, wins) = jax.lax.scan(
        step,
        (key, jnp.float32(100.0), jnp.float32(100.0)),
        jnp.arange(10)
    )
    return rewards.sum(), wins.mean()


def _winrate_vs_episode(theta, key, opponent_fn,
                        n_actions=N_ACTIONS, feature_fn=compute_features_batch_jax):
    """One greedy evaluation episode against a specified opponent."""
    def step(carry, t):
        key, b1, b2 = carry
        key, kv, k_opp = jax.random.split(key, 3)

        v = jax.random.uniform(kv, minval=V_MIN, maxval=V_MAX)
        cands = jnp.linspace(0.0, b1, n_actions)
        phis = feature_fn(v, t, b1, b2, cands)

        q_vals = phis @ theta

        idx = jnp.where(b1 <= 1e-10, jnp.int32(0), jnp.argmax(q_vals))
        bid1 = cands[idx]
        bid2 = opponent_fn(t, v, b2, b1, k_opp)

        reward = jnp.where(bid1 > bid2, v,
                 jnp.where(bid1 == bid2, v * 0.5, 0.0))
        won = jnp.float32(bid1 > bid2)

        return (key, b1 - bid1, b2 - bid2), (reward, won)

    _, (rewards, wins) = jax.lax.scan(
        step,
        (key, jnp.float32(100.0), jnp.float32(100.0)),
        jnp.arange(10)
    )
    return rewards.sum(), wins.mean()


# Public evaluation functions: single theta.

@partial(jit, static_argnames=('n_eval', 'n_actions', 'feature_fn'))
def evaluate_theta(theta, key, n_eval=500,
                   n_actions=N_ACTIONS, feature_fn=compute_features_batch_jax):
    """Evaluate a trained theta using greedy policy against default_player_2."""
    keys = jax.random.split(key, n_eval)
    return jax.vmap(
        lambda k: _eval_episode(theta, k, n_actions, feature_fn)
    )(keys).mean()


@partial(jit, static_argnames=('n_eval', 'n_actions', 'feature_fn'))
def evaluate_prediction(theta, key, n_eval=500,
                        n_actions=N_ACTIONS, feature_fn=compute_features_batch_jax):
    """Evaluate prediction accuracy: (actual_mean, predicted_mean)."""
    keys = jax.random.split(key, n_eval)
    actual, predicted = jax.vmap(
        lambda k: _pred_episode(theta, k, n_actions, feature_fn)
    )(keys)
    return actual.mean(), predicted.mean()


# Batch runners: vmap over checkpointed thetas.

def run_evaluation(checkpointed_thetas, label, eval_key, eval_every=1000,
                   n_actions=N_ACTIONS, feature_fn=compute_features_batch_jax, n_eval=500):
    """Evaluate checkpointed thetas for control (greedy policy evaluation)."""
    thetas_jax = jnp.array(checkpointed_thetas)
    keys = jax.random.split(eval_key, len(checkpointed_thetas))
    return np.array(jax.vmap(
        lambda t, k: evaluate_theta(t, k, n_eval=n_eval,
                                    n_actions=n_actions, feature_fn=feature_fn)
    )(thetas_jax, keys))


def run_prediction_evaluation(checkpointed_thetas, eval_key, eval_every=1000,
                               n_actions=N_ACTIONS, feature_fn=compute_features_batch_jax):
    """Evaluate checkpointed thetas for prediction (predicted vs actual)."""
    thetas_jax = jnp.array(checkpointed_thetas)
    keys = jax.random.split(eval_key, len(checkpointed_thetas))
    actual, predicted = jax.vmap(
        lambda t, k: evaluate_prediction(t, k, n_eval=500,
                                         n_actions=n_actions, feature_fn=feature_fn)
    )(thetas_jax, keys)
    return np.array(actual), np.array(predicted)


def run_winrate_evaluation(checkpointed_thetas, eval_key, eval_every=1000,
                           n_actions=N_ACTIONS, feature_fn=compute_features_batch_jax, n_eval=500):
    """Evaluate win rate for checkpointed thetas using the greedy policy."""
    thetas_jax = jnp.array(checkpointed_thetas)
    keys = jax.random.split(eval_key, len(checkpointed_thetas))

    def eval_one(theta, key):
        subkeys = jax.random.split(key, n_eval)
        _, win_rates = jax.vmap(
            lambda k: _winrate_episode(theta, k, n_actions, feature_fn)
        )(subkeys)
        return win_rates.mean()

    return np.array(jax.jit(jax.vmap(eval_one))(thetas_jax, keys))


def run_winrate_vs(checkpointed_thetas, eval_key, opponent_fn, eval_every=1000,
                   n_actions=N_ACTIONS, feature_fn=compute_features_batch_jax):
    """Evaluate win rate for checkpointed thetas against a specified opponent."""
    thetas_jax = jnp.array(checkpointed_thetas)
    keys = jax.random.split(eval_key, len(checkpointed_thetas))

    def eval_one(theta, key):
        subkeys = jax.random.split(key, 500)
        _, win_rates = jax.vmap(
            lambda k: _winrate_vs_episode(theta, k, opponent_fn, n_actions, feature_fn)
        )(subkeys)
        return win_rates.mean()

    return np.array(jax.vmap(eval_one)(thetas_jax, keys))
