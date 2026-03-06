"""SARSA agents: eligibility traces and n-step returns.

Contains:
- SARSA_lambda: SARSA with accumulating eligibility traces (Sutton Ch 12)
- N_step_SARSA: N-step semi-gradient SARSA (Sutton Ch 7)

Both use MSE loss only. The semi-gradient update bypasses the loss function
abstraction.
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from sections.episodes import mse, CLIP


# SARSA(lambda) with accumulating eligibility traces (Sutton Ch 12).

@partial(jit, static_argnums=(3, 4, 5, 7))
def _SARSA_lambda(phis, rewards, alphas, lam, gamma, eval_every, clip, reg,
                  theta_init):
    """JIT-compiled SARSA(lambda) core. Call the public SARSA_lambda() wrapper."""
    N = phis.shape[0]
    n_feats = phis.shape[-1]
    n_checkpoints = N // eval_every

    def one_episode(theta, ep_data):
        phi_ep, rew_ep, alpha = ep_data

        phi_next = jnp.concatenate([phi_ep[1:], jnp.zeros((1, n_feats))], axis=0)
        is_terminal = jnp.array([0.] * 9 + [1.])

        def forward(carry, inp):
            theta, e_trace = carry
            phi_t, phi_next_t, r, is_term = inp

            # TD error: delta = R + gamma*Q(S',A') - Q(S,A)
            q = jnp.dot(phi_t, theta)
            q_next = jnp.dot(phi_next_t, theta) * (1.0 - is_term)
            delta = r + gamma * q_next - q

            # Accumulating trace: e <- gamma*lambda*e + phi(S,A)
            e_trace = gamma * lam * e_trace + phi_t

            # Semi-gradient update: theta <- theta + alpha*delta*e
            update = delta * e_trace
            if reg:
                update_norm = jnp.linalg.norm(update)
                update = jnp.where(
                    update_norm > clip,
                    update * (clip / jnp.maximum(update_norm, 1e-8)),
                    update,
                )
            theta = theta + alpha * update
            return (theta, e_trace), delta ** 2

        (theta_new, _), losses = jax.lax.scan(
            forward,
            (theta, jnp.zeros(n_feats)),
            (phi_ep, phi_next, rew_ep, is_terminal)
        )
        return theta_new, losses.mean()

    def chunk_train(theta, chunk_data):
        phi_chunk, rew_chunk, alp_chunk = chunk_data
        theta_new, losses = jax.lax.scan(
            one_episode, theta, (phi_chunk, rew_chunk, alp_chunk)
        )
        return theta_new, (theta_new, losses.mean())

    _, (checkpointed_thetas, chunk_losses) = jax.lax.scan(
        chunk_train,
        theta_init,
        (
            phis.reshape(n_checkpoints, eval_every, 10, n_feats),
            rewards.reshape(n_checkpoints, eval_every, 10),
            alphas.reshape(n_checkpoints, eval_every),
        )
    )
    return checkpointed_thetas, chunk_losses


def SARSA_lambda(phis, rewards, alphas, lam=0.8, gamma=1.0, eval_every=1000,
                 clip=CLIP, reg=True, theta_init=None):
    """SARSA(lambda) with accumulating eligibility traces (Sutton Ch 12). """
    if theta_init is None:
        n_feats = phis.shape[-1]
        theta_init = jnp.zeros(n_feats)
    return _SARSA_lambda(phis, rewards, alphas, lam, gamma, eval_every, clip,
                         reg, theta_init)


# N-step semi-gradient SARSA (Sutton Ch 7).

@partial(jit, static_argnums=(3, 4, 5, 7))
def _N_step_SARSA(phis, rewards, alphas, n_step, gamma, eval_every, clip, reg,
                  theta_init):
    """JIT-compiled N-step SARSA core. Call the public N_step_SARSA() wrapper."""
    N = phis.shape[0]
    n_feats = phis.shape[-1]
    n_checkpoints = N // eval_every
    loss_and_grad = jax.value_and_grad(mse, argnums=0)

    def one_episode(theta, ep_data):
        phi_ep, rew_ep, alpha = ep_data

        # Extend phi array with a zero row for out-of-bounds bootstrap indexing.
        phi_ext = jnp.concatenate([phi_ep, jnp.zeros((1, n_feats))], axis=0)

        def forward(theta, t):
            # Compute n-step return: G = sum R[t:t+n] + Q(s_{t+n})
            end = jnp.minimum(t + n_step, 10)

            # Reward sum via masking (handles variable-length windows in JAX).
            step_indices = jnp.arange(10)
            mask = ((step_indices >= t) & (step_indices < end)).astype(jnp.float32)
            rew_sum = jnp.sum(rew_ep * mask)

            # Bootstrap Q(s_{t+n}) if t+n < T; else 0 (pure MC tail).
            bootstrap_phi = phi_ext[end]
            bootstrap_q = jnp.dot(bootstrap_phi, theta) * (end < 10).astype(jnp.float32)
            target = jax.lax.stop_gradient(rew_sum + bootstrap_q)

            # Semi-gradient MSE update.
            phi_t = phi_ep[t]
            loss, grad = loss_and_grad(theta, phi_t, target)
            if reg:
                grad_norm = jnp.linalg.norm(grad)
                grad = jnp.where(
                    grad_norm > clip,
                    grad * (clip / jnp.maximum(grad_norm, 1e-8)),
                    grad,
                )
            theta = theta - alpha * grad
            return theta, loss

        theta_new, losses = jax.lax.scan(forward, theta, jnp.arange(10))
        return theta_new, losses.mean()

    def chunk_train(theta, chunk_data):
        phi_chunk, rew_chunk, alp_chunk = chunk_data
        theta_new, losses = jax.lax.scan(
            one_episode, theta, (phi_chunk, rew_chunk, alp_chunk)
        )
        return theta_new, (theta_new, losses.mean())

    _, (checkpointed_thetas, chunk_losses) = jax.lax.scan(
        chunk_train,
        theta_init,
        (
            phis.reshape(n_checkpoints, eval_every, 10, n_feats),
            rewards.reshape(n_checkpoints, eval_every, 10),
            alphas.reshape(n_checkpoints, eval_every),
        )
    )
    return checkpointed_thetas, chunk_losses


def N_step_SARSA(phis, rewards, alphas, n_step=4, gamma=1.0, eval_every=1000,
                 clip=CLIP, reg=True, theta_init=None):
    """N-step semi-gradient SARSA (Sutton Ch 7). """
    if theta_init is None:
        n_feats = phis.shape[-1]
        theta_init = jnp.zeros(n_feats)
    return _N_step_SARSA(phis, rewards, alphas, n_step, gamma, eval_every, clip,
                         reg, theta_init)
