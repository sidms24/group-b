"""Batch Monte Carlo agent.

All 10 timestep gradients within an episode are computed against the same
theta and averaged into a single update per episode.  This avoids within-
episode theta drift that occurs with online (per-step) updates.

Supports mini-batch SGD: when batch_size > 1, multiple episodes share the
same frozen theta and their parameter displacements are averaged before
applying a single update.  This reduces gradient variance at the cost of
fewer updates per data (N/batch_size updates instead of N).

Contains:
- mc_target: compute scalar MC target (cumulative return)
- _MC: JIT-compiled batch MC training core
- MC: public wrapper with default initialisation
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from sections.episodes import CLIP


def mc_target(G):
    """Compute MC target: the scalar cumulative return G."""
    return G


@partial(jit, static_argnums=(3, 4, 6, 10))
def _MC(phis, rewards, alphas, loss_fn, eval_every, clip, reg, theta_init,
        lambda_l1=0.0, lambda_l2=0.0, batch_size=1):
    """JIT-compiled batch MC core. Call the public MC() wrapper instead."""
    N = phis.shape[0]
    n_feats = phis.shape[-1]
    n_checkpoints = N // eval_every
    loss_and_grad = jax.value_and_grad(loss_fn, argnums=0)

    def one_episode(theta, ep_data):
        phi_ep, rew_ep, alpha = ep_data

        # Backward scan to accumulate returns only (theta is not modified).
        def accumulate_returns(G, r):
            G = r + G
            return G, G

        _, returns = jax.lax.scan(
            accumulate_returns,
            jnp.float32(0.0),
            rew_ep[::-1],
        )
        returns = returns[::-1]  # G_0, G_1, ..., G_9

        # Compute gradients for all 10 timesteps against the same theta.
        def step_loss_and_grad(inp):
            phi, target = inp
            return loss_and_grad(theta, phi, target)

        losses, grads = jax.vmap(step_loss_and_grad)((phi_ep, returns))

        # Average gradient across timesteps, then apply regularisation.
        mean_grad = grads.mean(axis=0)
        mean_grad = mean_grad + lambda_l2 * theta + lambda_l1 * jnp.sign(theta)
        if reg:
            grad_norm = jnp.linalg.norm(mean_grad)
            mean_grad = jnp.where(
                grad_norm > clip,
                mean_grad * (clip / jnp.maximum(grad_norm, 1e-8)),
                mean_grad,
            )

        theta_new = theta - alpha * mean_grad
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


def MC(phis, rewards, alphas, loss_fn, eval_every=1000, clip=CLIP, reg=True,
       theta_init=None, lambda_l1=0.0, lambda_l2=0.0, batch_size=1):
    """Batch Monte Carlo training with optional mini-batch SGD. """
    assert eval_every % batch_size == 0, \
        f'eval_every ({eval_every}) must be divisible by batch_size ({batch_size})'
    if theta_init is None:
        n_feats = phis.shape[-1]
        theta_init = jnp.zeros(n_feats)
    return _MC(phis, rewards, alphas, loss_fn, eval_every, clip, reg, theta_init,
               lambda_l1, lambda_l2, batch_size)
