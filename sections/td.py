"""Batch Temporal Difference agent with forward-scan updates.

All 10 timestep updates within an episode use a forward scan where theta
evolves step-by-step (TD bootstrapping requires this).

Supports mini-batch SGD: when batch_size > 1, multiple episodes share the
same frozen theta and their parameter displacements are averaged before
applying a single update.  This reduces gradient variance at the cost of
fewer updates per data (N/batch_size updates instead of N).

Contains:
- td_target: compute scalar TD target (r + gamma * Q(s', a'))
- _TD: JIT-compiled batch TD training core
- TD: public wrapper with default initialisation
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from sections.episodes import CLIP


def td_target(r, phi_next, theta, gamma, is_terminal):
    """Compute TD target: scalar r + gamma * Q(s', a') * (1 - terminal)."""
    q_next = jnp.dot(theta, phi_next)
    return r + gamma * q_next * (1.0 - is_terminal)


# TD training applies a forward scan over each episode step.

@partial(jit, static_argnums=(3, 4, 5, 7, 11))
def _TD(phis, rewards, alphas, loss_fn, gamma, eval_every, clip, reg, theta_init,
        lambda_l1=0.0, lambda_l2=0.0, batch_size=1):
    """JIT-compiled batch TD core. Call the public TD() wrapper instead."""
    N = phis.shape[0]
    n_feats = phis.shape[-1]
    n_checkpoints = N // eval_every
    loss_and_grad = jax.value_and_grad(loss_fn, argnums=0)

    def one_episode(theta, ep_data):
        phi_ep, rew_ep, alpha = ep_data

        phi_next = jnp.concatenate([phi_ep[1:], jnp.zeros((1, n_feats))], axis=0)
        is_terminal = jnp.array([0.] * 9 + [1.])

        def forward(theta, inp):
            phi_t, phi_next_t, r, is_term = inp
            target = td_target(r, phi_next_t, theta, gamma, is_term)
            td_err, grad = loss_and_grad(theta, phi_t, target)
            # L2 penalty (weight decay) and L1 penalty (sparsity).
            grad = grad + lambda_l2 * theta + lambda_l1 * jnp.sign(theta)
            if reg:
                grad_norm = jnp.linalg.norm(grad)
                grad = jnp.where(
                    grad_norm > clip,
                    grad * (clip / jnp.maximum(grad_norm, 1e-8)),
                    grad,
                )
            theta = theta - alpha * grad
            return theta, td_err

        theta_new, losses = jax.lax.scan(
            forward,
            theta,
            (phi_ep, phi_next, rew_ep, is_terminal)
        )
        return theta_new, losses.mean()

    def batch_update(theta, batch_data):
        """Run batch_size episodes from same theta, average displacement."""
        phi_batch, rew_batch, alp_batch = batch_data
        theta_news, losses = jax.vmap(
            lambda p, r, a: one_episode(theta, (p, r, a))
        )(phi_batch, rew_batch, alp_batch)
        mean_theta = theta + (theta_news - theta).mean(axis=0)
        return mean_theta, losses.mean()

    def chunk_train(theta, chunk_data):
        phi_chunk, rew_chunk, alp_chunk = chunk_data
        n_batches = eval_every // batch_size
        theta_new, losses = jax.lax.scan(
            batch_update, theta,
            (phi_chunk.reshape(n_batches, batch_size, 10, n_feats),
             rew_chunk.reshape(n_batches, batch_size, 10),
             alp_chunk.reshape(n_batches, batch_size))
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


def TD(phis, rewards, alphas, loss_fn, gamma=1.0, eval_every=1000, clip=CLIP,
       reg=True, theta_init=None, lambda_l1=0.0, lambda_l2=0.0, batch_size=1):
    """Batch Temporal Difference training with forward scan. """
    assert eval_every % batch_size == 0, \
        f'eval_every ({eval_every}) must be divisible by batch_size ({batch_size})'
    if theta_init is None:
        n_feats = phis.shape[-1]
        theta_init = jnp.zeros(n_feats)
    return _TD(phis, rewards, alphas, loss_fn, gamma, eval_every, clip, reg, theta_init,
               lambda_l1, lambda_l2, batch_size)
