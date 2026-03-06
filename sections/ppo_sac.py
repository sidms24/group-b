"""PPO-SAC: Proximal Policy Optimization with SAC-inspired entropy tuning.

Combines PPO's stable clipped surrogate objective with:
- Twin value critics V1(s), V2(s) for conservative advantage estimation
- Automatic entropy temperature alpha (from SAC) for adaptive exploration
- Categorical policy over discrete bid actions
- GAE(lambda) advantage estimation
- Self-play training against frozen opponent

Architecture:
  state_features -> actor head (softmax over n_actions)
  state_features -> twin critic head (V1(s), V2(s))
  learnable log_alpha -> entropy temperature
"""

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
from functools import partial
from tqdm import tqdm

from sections.policies import default_player_2


# ── State feature functions (reuse from actor_critic) ──

def raw_state_features(v, t, own_b, opp_b):
    """Minimal normalized state features. Returns (4,)."""
    v = jnp.asarray(v, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)
    own_b = jnp.asarray(own_b, dtype=jnp.float32)
    opp_b = jnp.asarray(opp_b, dtype=jnp.float32)
    return jnp.array([
        (v - 10.0) / 10.0,
        t / 9.0,
        own_b / 100.0,
        opp_b / 100.0,
    ])


N_STATE_RAW = 4


def binned_state_features(v, t, own_b, opp_b):
    """Binned state features (one-hot bins + continuous). Returns (41,)."""
    v = jnp.asarray(v, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)
    own_b = jnp.asarray(own_b, dtype=jnp.float32)
    opp_b = jnp.asarray(opp_b, dtype=jnp.float32)

    v_idx = jnp.clip(jnp.floor(v - 10.0).astype(jnp.int32), 0, 9)
    t_idx = jnp.clip((t / 2).astype(jnp.int32), 0, 4)
    ob_idx = jnp.clip(jnp.floor(own_b / 10.0).astype(jnp.int32), 0, 9)
    op_idx = jnp.clip(jnp.floor(opp_b / 10.0).astype(jnp.int32), 0, 9)
    budget_ratio = own_b / jnp.maximum(own_b + opp_b, 1e-8)

    return jnp.concatenate([
        jax.nn.one_hot(v_idx, 10),
        jax.nn.one_hot(t_idx, 5),
        jax.nn.one_hot(ob_idx, 10),
        jax.nn.one_hot(op_idx, 10),
        jnp.array([
            (own_b - opp_b) / 100.0,
            (v - 10.0) / 10.0,
            t / 9.0,
            own_b / 100.0,
            opp_b / 100.0,
            budget_ratio,
        ]),
    ])


N_STATE_BINNED = 41


def make_tile_state_features(n_tilings=4, n_tiles=8):
    """Tile-coded state features."""
    n_features = n_tilings * 2 * n_tiles * n_tiles
    offsets = jnp.linspace(0, 1.0 / n_tiles, n_tilings, endpoint=False)

    def feature_fn(v, t, own_b, opp_b):
        v = jnp.asarray(v, dtype=jnp.float32)
        t = jnp.asarray(t, dtype=jnp.float32)
        own_b = jnp.asarray(own_b, dtype=jnp.float32)
        opp_b = jnp.asarray(opp_b, dtype=jnp.float32)

        v_norm = jnp.clip((v - 10.0) / 10.0, 0.0, 0.999)
        t_norm = jnp.clip(t / 9.0, 0.0, 0.999)
        br = jnp.clip(own_b / jnp.maximum(own_b + opp_b, 1e-8), 0.0, 0.999)

        tiles_per_pair = n_tiles * n_tiles
        out = jnp.zeros(n_features)

        for i in range(n_tilings):
            off = offsets[i]
            vi = jnp.clip(jnp.floor((v_norm + off) * n_tiles).astype(jnp.int32), 0, n_tiles - 1)
            bi = jnp.clip(jnp.floor((br + off) * n_tiles).astype(jnp.int32), 0, n_tiles - 1)
            idx1 = i * 2 * tiles_per_pair + vi * n_tiles + bi
            out = out.at[idx1].set(1.0)

            ti = jnp.clip(jnp.floor((t_norm + off) * n_tiles).astype(jnp.int32), 0, n_tiles - 1)
            idx2 = i * 2 * tiles_per_pair + tiles_per_pair + ti * n_tiles + bi
            out = out.at[idx2].set(1.0)

        return out

    return feature_fn, n_features


# ── Network architectures ──

class PolicyNetwork(nn.Module):
    """Categorical policy: state -> logits over n_actions bid levels."""
    n_actions: int = 51
    hidden: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        return nn.Dense(self.n_actions)(x)


class TwinValueNetwork(nn.Module):
    """Twin value networks: state -> (V1, V2) for conservative estimation."""
    hidden: int = 256

    @nn.compact
    def __call__(self, x):
        # V1
        h1 = nn.Dense(self.hidden, name='v1_dense1')(x)
        h1 = nn.relu(h1)
        h1 = nn.Dense(self.hidden, name='v1_dense2')(h1)
        h1 = nn.relu(h1)
        v1 = nn.Dense(1, name='v1_out')(h1).squeeze(-1)

        # V2
        h2 = nn.Dense(self.hidden, name='v2_dense1')(x)
        h2 = nn.relu(h2)
        h2 = nn.Dense(self.hidden, name='v2_dense2')(h2)
        h2 = nn.relu(h2)
        v2 = nn.Dense(1, name='v2_out')(h2).squeeze(-1)

        return v1, v2


# ── Episode generation ──

@partial(jax.jit, static_argnames=(
    'n_episodes', 'n_actions', 'actor_apply', 'critic_apply', 'state_fn',
))
def generate_episodes_ppo(
    n_episodes, key, actor_params, critic_params,
    actor_apply, critic_apply,
    n_actions=51, state_fn=None,
):
    """Generate episodes for PPO training.

    Returns:
        state_feats: (n_episodes, 10, n_state_features)
        actions:     (n_episodes, 10) int32
        rewards:     (n_episodes, 10)
        log_probs:   (n_episodes, 10) log pi(a|s)
        values:      (n_episodes, 10) min(V1, V2)
    """
    _state_fn = state_fn or raw_state_features

    def one_episode(key):
        def step(carry, t):
            key, b1, b2 = carry
            key, kv, kn, ka = jax.random.split(key, 4)

            v = jax.random.uniform(kv, minval=10.0, maxval=20.0)
            s_feat = _state_fn(v, jnp.float32(t), b1, b2)

            logits = actor_apply(actor_params, s_feat)
            v1, v2 = critic_apply(critic_params, s_feat)
            value = jnp.minimum(v1, v2)

            log_pi = jax.nn.log_softmax(logits)
            action_idx = jax.random.categorical(ka, logits)
            log_prob = log_pi[action_idx]

            cands = jnp.linspace(0.0, b1, n_actions)
            bid1 = cands[action_idx]

            bid2 = default_player_2(t, v, b2, b1, kn)
            reward = jnp.where(bid1 > bid2, v,
                     jnp.where(bid1 == bid2, v * 0.5, 0.0))

            return (key, b1 - bid1, b2 - bid2), (s_feat, action_idx, reward, log_prob, value)

        _, outputs = jax.lax.scan(
            step,
            (key, jnp.float32(100.0), jnp.float32(100.0)),
            jnp.arange(10),
        )
        return outputs

    keys = jax.random.split(key, n_episodes)
    return jax.lax.map(one_episode, keys)


@partial(jax.jit, static_argnames=(
    'n_episodes', 'n_actions', 'actor_apply', 'critic_apply', 'state_fn',
))
def generate_episodes_ppo_selfplay(
    n_episodes, key, actor_params, critic_params,
    opponent_actor_params,
    actor_apply, critic_apply,
    n_actions=51, state_fn=None,
):
    """Generate episodes: PPO agent vs frozen opponent actor (greedy)."""
    _state_fn = state_fn or raw_state_features

    def one_episode(key):
        def step(carry, t):
            key, b1, b2 = carry
            key, kv, ka = jax.random.split(key, 3)

            v = jax.random.uniform(kv, minval=10.0, maxval=20.0)

            # Player 1: sample from policy
            s_feat1 = _state_fn(v, jnp.float32(t), b1, b2)
            logits = actor_apply(actor_params, s_feat1)
            v1, v2 = critic_apply(critic_params, s_feat1)
            value = jnp.minimum(v1, v2)

            log_pi = jax.nn.log_softmax(logits)
            action_idx = jax.random.categorical(ka, logits)
            log_prob = log_pi[action_idx]

            cands1 = jnp.linspace(0.0, b1, n_actions)
            bid1 = cands1[action_idx]

            # Player 2: greedy from opponent
            s_feat2 = _state_fn(v, jnp.float32(t), b2, b1)
            opp_logits = actor_apply(opponent_actor_params, s_feat2)
            opp_idx = jnp.where(b2 <= 1e-10, jnp.int32(0), jnp.argmax(opp_logits))
            cands2 = jnp.linspace(0.0, b2, n_actions)
            bid2 = cands2[opp_idx]

            reward = jnp.where(bid1 > bid2, v,
                     jnp.where(bid1 == bid2, v * 0.5, 0.0))

            return (key, b1 - bid1, b2 - bid2), (s_feat1, action_idx, reward, log_prob, value)

        _, outputs = jax.lax.scan(
            step,
            (key, jnp.float32(100.0), jnp.float32(100.0)),
            jnp.arange(10),
        )
        return outputs

    keys = jax.random.split(key, n_episodes)
    return jax.lax.map(one_episode, keys)


# ── Advantage computation ──

def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """GAE(lambda) with terminal V=0."""
    T = rewards.shape[1]
    values_ext = jnp.concatenate([values, jnp.zeros_like(values[:, :1])], axis=1)

    advantages = jnp.zeros_like(rewards)
    gae = jnp.zeros(rewards.shape[0])

    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * values_ext[:, t + 1] - values[:, t]
        gae = delta + gamma * lam * gae
        advantages = advantages.at[:, t].set(gae)

    returns = advantages + values
    return advantages, returns


# ── PPO training step ──

@partial(jax.jit, static_argnames=(
    'actor_apply', 'critic_apply', 'actor_opt', 'critic_opt', 'alpha_opt',
))
def ppo_train_step(
    actor_params, critic_params, log_alpha,
    actor_opt_state, critic_opt_state, alpha_opt_state,
    actor_apply, critic_apply,
    actor_opt, critic_opt, alpha_opt,
    states, actions, returns, advantages, log_probs_old,
    clip_ratio=0.2, target_entropy=-1.0,
):
    """One PPO gradient step with auto entropy temperature.

    Args:
        log_alpha: learnable log entropy temperature
        clip_ratio: PPO clipping parameter
        target_entropy: target entropy for alpha tuning (negative)
    """
    n_actions = actions.max() + 1  # inferred from data
    adv = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    alpha = jnp.exp(log_alpha)

    # ── Actor loss (PPO clipped + entropy) ──
    def actor_loss_fn(a_params):
        logits = jax.vmap(actor_apply, in_axes=(None, 0))(a_params, states)
        log_pi = jax.nn.log_softmax(logits)
        pi = jax.nn.softmax(logits)
        log_pi_a = log_pi[jnp.arange(states.shape[0]), actions]

        # Importance ratio
        ratio = jnp.exp(log_pi_a - log_probs_old)

        # Clipped surrogate
        adv_stopped = jax.lax.stop_gradient(adv)
        surr1 = ratio * adv_stopped
        surr2 = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_stopped
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        # Entropy bonus (weighted by alpha)
        entropy = -jnp.mean(jnp.sum(pi * log_pi, axis=1))
        total_loss = policy_loss - jax.lax.stop_gradient(alpha) * entropy

        return total_loss, (entropy, jnp.mean(jnp.abs(ratio - 1.0)))

    (a_loss, (entropy, ratio_dev)), a_grads = jax.value_and_grad(
        actor_loss_fn, has_aux=True
    )(actor_params)
    a_updates, actor_opt_state = actor_opt.update(a_grads, actor_opt_state, actor_params)
    actor_params = optax.apply_updates(actor_params, a_updates)

    # ── Critic loss (twin MSE) ──
    def critic_loss_fn(c_params):
        v1, v2 = jax.vmap(critic_apply, in_axes=(None, 0))(c_params, states)
        loss1 = jnp.mean((v1 - returns) ** 2)
        loss2 = jnp.mean((v2 - returns) ** 2)
        return 0.5 * (loss1 + loss2)

    c_loss, c_grads = jax.value_and_grad(critic_loss_fn)(critic_params)
    c_updates, critic_opt_state = critic_opt.update(c_grads, critic_opt_state, critic_params)
    critic_params = optax.apply_updates(critic_params, c_updates)

    # ── Alpha loss (auto temperature) ──
    def alpha_loss_fn(la):
        return -jnp.exp(la) * jax.lax.stop_gradient(entropy + target_entropy)

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(log_alpha)
    alpha_update, alpha_opt_state = alpha_opt.update(alpha_grad, alpha_opt_state)
    log_alpha = optax.apply_updates(log_alpha, alpha_update)

    return (actor_params, critic_params, log_alpha,
            actor_opt_state, critic_opt_state, alpha_opt_state,
            a_loss, c_loss, entropy, jnp.exp(log_alpha))


# ── Evaluation ──

@partial(jax.jit, static_argnames=('n_eval', 'n_actions', 'actor_apply', 'state_fn'))
def evaluate_ppo(actor_params, key, actor_apply, n_eval=2000, n_actions=51, state_fn=None):
    """Evaluate PPO agent greedily against default_player_2."""
    _state_fn = state_fn or raw_state_features

    def one_eval(key):
        def step(carry, t):
            key, b1, b2, total = carry
            key, kv, kn = jax.random.split(key, 3)

            v = jax.random.uniform(kv, minval=10.0, maxval=20.0)
            s_feat = _state_fn(v, jnp.float32(t), b1, b2)

            logits = actor_apply(actor_params, s_feat)
            idx = jnp.where(b1 <= 1e-10, jnp.int32(0), jnp.argmax(logits))

            cands = jnp.linspace(0.0, b1, n_actions)
            bid1 = cands[idx]
            bid2 = default_player_2(t, v, b2, b1, kn)

            reward = jnp.where(bid1 > bid2, v,
                     jnp.where(bid1 == bid2, v * 0.5, 0.0))
            return (key, b1 - bid1, b2 - bid2, total + reward), None

        (_, _, _, total), _ = jax.lax.scan(
            step,
            (key, jnp.float32(100.0), jnp.float32(100.0), jnp.float32(0.0)),
            jnp.arange(10),
        )
        return total

    keys = jax.random.split(key, n_eval)
    return jnp.mean(jax.vmap(one_eval)(keys))


# ── Experiment runners ──

def run_ppo_experiment(
    N=200_000,
    eval_every=1000,
    n_actions=51,
    lr_actor=3e-4,
    lr_critic=1e-3,
    lr_alpha=3e-4,
    hidden=256,
    n_epochs=4,
    batch_size=2048,
    n_eval=2000,
    seed=24,
    state_fn=None,
    n_state_features=4,
    clip_ratio=0.2,
    gae_lambda=0.95,
    target_entropy=None,
    max_grad_norm=0.5,
):
    """Train a PPO-SAC agent against the default heuristic.

    Args:
        n_epochs: PPO epochs per rollout batch
        clip_ratio: PPO clipping epsilon
        target_entropy: SAC target entropy; defaults to -0.5 * log(n_actions)
        max_grad_norm: gradient clipping norm

    Returns:
        dict with rewards, losses, params, best_reward
    """
    if target_entropy is None:
        target_entropy = -0.5 * np.log(n_actions)

    actor = PolicyNetwork(n_actions=n_actions, hidden=hidden)
    critic = TwinValueNetwork(hidden=hidden)

    key = jax.random.PRNGKey(seed)
    key, ak, ck = jax.random.split(key, 3)
    dummy = jnp.zeros(n_state_features)
    actor_params = actor.init(ak, dummy)
    critic_params = critic.init(ck, dummy)
    log_alpha = jnp.array(0.0)

    actor_opt = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_actor))
    critic_opt = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_critic))
    alpha_opt = optax.adam(lr_alpha)

    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)
    alpha_opt_state = alpha_opt.init(log_alpha)

    n_chunks = N // eval_every
    best_actor_params = actor_params
    best_critic_params = critic_params
    best_reward = -np.inf
    all_rewards, all_actor_losses, all_critic_losses, all_entropy, all_alpha = [], [], [], [], []

    pbar = tqdm(range(n_chunks), desc='PPO-SAC')
    for chunk in pbar:
        key, gen_key, eval_key, shuf_key = jax.random.split(key, 4)

        # Collect episodes
        state_feats, actions, rewards, log_probs, values = generate_episodes_ppo(
            eval_every, gen_key, actor_params, critic_params,
            actor.apply, critic.apply,
            n_actions=n_actions, state_fn=state_fn,
        )

        # Compute GAE advantages
        advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=gae_lambda)

        # Flatten
        sf_flat = state_feats.reshape(-1, n_state_features)
        act_flat = actions.reshape(-1)
        ret_flat = returns.reshape(-1)
        adv_flat = advantages.reshape(-1)
        lp_flat = log_probs.reshape(-1)
        n_samples = sf_flat.shape[0]

        # PPO epochs
        chunk_a_loss, chunk_c_loss, chunk_ent, chunk_alpha = 0.0, 0.0, 0.0, 0.0
        n_steps = 0
        for epoch in range(n_epochs):
            shuf_key, perm_key = jax.random.split(shuf_key)
            perm = jax.random.permutation(perm_key, n_samples)

            for start in range(0, n_samples - batch_size + 1, batch_size):
                mb = perm[start:start + batch_size]

                (actor_params, critic_params, log_alpha,
                 actor_opt_state, critic_opt_state, alpha_opt_state,
                 a_loss, c_loss, ent, alpha_val) = ppo_train_step(
                    actor_params, critic_params, log_alpha,
                    actor_opt_state, critic_opt_state, alpha_opt_state,
                    actor.apply, critic.apply,
                    actor_opt, critic_opt, alpha_opt,
                    sf_flat[mb], act_flat[mb], ret_flat[mb], adv_flat[mb], lp_flat[mb],
                    clip_ratio=clip_ratio, target_entropy=target_entropy,
                )
                chunk_a_loss += float(a_loss)
                chunk_c_loss += float(c_loss)
                chunk_ent += float(ent)
                chunk_alpha += float(alpha_val)
                n_steps += 1

        # Evaluate
        chunk_reward = float(evaluate_ppo(
            actor_params, eval_key, actor.apply,
            n_eval=n_eval, n_actions=n_actions, state_fn=state_fn,
        ))

        if chunk_reward > best_reward:
            best_reward = chunk_reward
            best_actor_params = jax.tree.map(lambda x: x.copy(), actor_params)
            best_critic_params = jax.tree.map(lambda x: x.copy(), critic_params)

        all_rewards.append(chunk_reward)
        n_steps = max(n_steps, 1)
        all_actor_losses.append(chunk_a_loss / n_steps)
        all_critic_losses.append(chunk_c_loss / n_steps)
        all_entropy.append(chunk_ent / n_steps)
        all_alpha.append(chunk_alpha / n_steps)

        pbar.set_postfix(
            reward=f'{chunk_reward:.1f}', best=f'{best_reward:.1f}',
            alpha=f'{chunk_alpha / n_steps:.3f}', ent=f'{chunk_ent / n_steps:.2f}',
        )

    return {
        'rewards': np.array(all_rewards),
        'actor_losses': np.array(all_actor_losses),
        'critic_losses': np.array(all_critic_losses),
        'entropy': np.array(all_entropy),
        'alpha': np.array(all_alpha),
        'actor_params_best': best_actor_params,
        'critic_params_best': best_critic_params,
        'actor_params_final': actor_params,
        'critic_params_final': critic_params,
        'best_reward': best_reward,
    }


def run_ppo_selfplay_experiment(
    N=200_000,
    eval_every=1000,
    n_actions=51,
    lr_actor=3e-4,
    lr_critic=1e-3,
    lr_alpha=3e-4,
    hidden=256,
    n_epochs=4,
    batch_size=2048,
    n_eval=2000,
    seed=24,
    state_fn=None,
    n_state_features=4,
    clip_ratio=0.2,
    gae_lambda=0.95,
    target_entropy=None,
    max_grad_norm=0.5,
    opponent_update_every=10,
):
    """Train PPO-SAC via self-play against frozen opponent."""
    if target_entropy is None:
        target_entropy = -0.5 * np.log(n_actions)

    actor = PolicyNetwork(n_actions=n_actions, hidden=hidden)
    critic = TwinValueNetwork(hidden=hidden)

    key = jax.random.PRNGKey(seed)
    key, ak, ck, ok = jax.random.split(key, 4)
    dummy = jnp.zeros(n_state_features)
    actor_params = actor.init(ak, dummy)
    critic_params = critic.init(ck, dummy)
    opponent_actor_params = actor.init(ok, dummy)
    log_alpha = jnp.array(0.0)

    actor_opt = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_actor))
    critic_opt = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_critic))
    alpha_opt = optax.adam(lr_alpha)

    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)
    alpha_opt_state = alpha_opt.init(log_alpha)

    n_chunks = N // eval_every
    best_actor_params = actor_params
    best_critic_params = critic_params
    best_reward = -np.inf
    all_rewards, all_actor_losses, all_critic_losses = [], [], []
    all_entropy, all_alpha, selfplay_rewards = [], [], []

    pbar = tqdm(range(n_chunks), desc='PPO-SAC self-play')
    for chunk in pbar:
        key, gen_key, eval_key, shuf_key = jax.random.split(key, 4)

        # Update opponent periodically
        if chunk > 0 and chunk % opponent_update_every == 0:
            opponent_actor_params = jax.tree.map(lambda x: x.copy(), actor_params)

        # Collect self-play episodes
        state_feats, actions, rewards, log_probs, values = generate_episodes_ppo_selfplay(
            eval_every, gen_key, actor_params, critic_params,
            opponent_actor_params,
            actor.apply, critic.apply,
            n_actions=n_actions, state_fn=state_fn,
        )

        sp_reward = float(jnp.mean(jnp.sum(rewards, axis=1)))
        selfplay_rewards.append(sp_reward)

        # GAE
        advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=gae_lambda)

        # Flatten
        sf_flat = state_feats.reshape(-1, n_state_features)
        act_flat = actions.reshape(-1)
        ret_flat = returns.reshape(-1)
        adv_flat = advantages.reshape(-1)
        lp_flat = log_probs.reshape(-1)
        n_samples = sf_flat.shape[0]

        # PPO epochs
        chunk_a_loss, chunk_c_loss, chunk_ent, chunk_alpha = 0.0, 0.0, 0.0, 0.0
        n_steps = 0
        for epoch in range(n_epochs):
            shuf_key, perm_key = jax.random.split(shuf_key)
            perm = jax.random.permutation(perm_key, n_samples)

            for start in range(0, n_samples - batch_size + 1, batch_size):
                mb = perm[start:start + batch_size]

                (actor_params, critic_params, log_alpha,
                 actor_opt_state, critic_opt_state, alpha_opt_state,
                 a_loss, c_loss, ent, alpha_val) = ppo_train_step(
                    actor_params, critic_params, log_alpha,
                    actor_opt_state, critic_opt_state, alpha_opt_state,
                    actor.apply, critic.apply,
                    actor_opt, critic_opt, alpha_opt,
                    sf_flat[mb], act_flat[mb], ret_flat[mb], adv_flat[mb], lp_flat[mb],
                    clip_ratio=clip_ratio, target_entropy=target_entropy,
                )
                chunk_a_loss += float(a_loss)
                chunk_c_loss += float(c_loss)
                chunk_ent += float(ent)
                chunk_alpha += float(alpha_val)
                n_steps += 1

        # Evaluate against heuristic
        chunk_reward = float(evaluate_ppo(
            actor_params, eval_key, actor.apply,
            n_eval=n_eval, n_actions=n_actions, state_fn=state_fn,
        ))

        if chunk_reward > best_reward:
            best_reward = chunk_reward
            best_actor_params = jax.tree.map(lambda x: x.copy(), actor_params)
            best_critic_params = jax.tree.map(lambda x: x.copy(), critic_params)

        all_rewards.append(chunk_reward)
        n_steps = max(n_steps, 1)
        all_actor_losses.append(chunk_a_loss / n_steps)
        all_critic_losses.append(chunk_c_loss / n_steps)
        all_entropy.append(chunk_ent / n_steps)
        all_alpha.append(chunk_alpha / n_steps)

        pbar.set_postfix(
            reward=f'{chunk_reward:.1f}', best=f'{best_reward:.1f}',
            sp_r=f'{sp_reward:.1f}', alpha=f'{chunk_alpha / n_steps:.3f}',
        )

    return {
        'rewards': np.array(all_rewards),
        'selfplay_rewards': np.array(selfplay_rewards),
        'actor_losses': np.array(all_actor_losses),
        'critic_losses': np.array(all_critic_losses),
        'entropy': np.array(all_entropy),
        'alpha': np.array(all_alpha),
        'actor_params_best': best_actor_params,
        'critic_params_best': best_critic_params,
        'actor_params_final': actor_params,
        'critic_params_final': critic_params,
        'best_reward': best_reward,
    }


# ── Discrete SAC ──

class TwinQNetwork(nn.Module):
    """Twin Q-networks for discrete SAC: state -> (Q1[all_actions], Q2[all_actions])."""
    n_actions: int = 51
    hidden: int = 256

    @nn.compact
    def __call__(self, x):
        h1 = nn.Dense(self.hidden, name='q1_d1')(x)
        h1 = nn.relu(h1)
        h1 = nn.Dense(self.hidden, name='q1_d2')(h1)
        h1 = nn.relu(h1)
        q1 = nn.Dense(self.n_actions, name='q1_out')(h1)

        h2 = nn.Dense(self.hidden, name='q2_d1')(x)
        h2 = nn.relu(h2)
        h2 = nn.Dense(self.hidden, name='q2_d2')(h2)
        h2 = nn.relu(h2)
        q2 = nn.Dense(self.n_actions, name='q2_out')(h2)

        return q1, q2


@partial(jax.jit, static_argnames=('actor_apply', 'q_apply', 'actor_opt', 'q_opt', 'alpha_opt'))
def sac_train_step(
    actor_params, q_params, q_target_params, log_alpha,
    actor_opt_state, q_opt_state, alpha_opt_state,
    actor_apply, q_apply,
    actor_opt, q_opt, alpha_opt,
    states, actions, rewards, next_states, dones,
    gamma=1.0, target_entropy=-1.0, tau=0.005,
):
    """One discrete SAC gradient step.

    Args:
        states:      (batch, n_features)
        actions:     (batch,) int32
        rewards:     (batch,)
        next_states: (batch, n_features)
        dones:       (batch,) bool/float
        gamma:       discount factor
        target_entropy: target entropy for alpha
        tau:         soft target update rate
    """
    alpha = jnp.exp(log_alpha)
    batch_size = states.shape[0]

    # ── Q loss ──
    # Target: r + gamma * (1 - done) * (sum_a pi(a|s') * (min(Q1_targ, Q2_targ) - alpha * log pi(a|s')))
    next_logits = jax.vmap(actor_apply, in_axes=(None, 0))(actor_params, next_states)
    next_pi = jax.nn.softmax(next_logits)
    next_log_pi = jax.nn.log_softmax(next_logits)

    q1_targ, q2_targ = jax.vmap(q_apply, in_axes=(None, 0))(q_target_params, next_states)
    min_q_targ = jnp.minimum(q1_targ, q2_targ)

    v_next = jnp.sum(next_pi * (min_q_targ - alpha * next_log_pi), axis=1)
    q_target = rewards + gamma * (1.0 - dones) * v_next

    def q_loss_fn(q_p):
        q1, q2 = jax.vmap(q_apply, in_axes=(None, 0))(q_p, states)
        q1_a = q1[jnp.arange(batch_size), actions]
        q2_a = q2[jnp.arange(batch_size), actions]
        target = jax.lax.stop_gradient(q_target)
        return 0.5 * (jnp.mean((q1_a - target) ** 2) + jnp.mean((q2_a - target) ** 2))

    q_loss, q_grads = jax.value_and_grad(q_loss_fn)(q_params)
    q_updates, q_opt_state = q_opt.update(q_grads, q_opt_state, q_params)
    q_params = optax.apply_updates(q_params, q_updates)

    # ── Actor loss ──
    # Maximize: E_a~pi [min(Q1, Q2)(s,a) - alpha * log pi(a|s)]
    def actor_loss_fn(a_p):
        logits = jax.vmap(actor_apply, in_axes=(None, 0))(a_p, states)
        pi = jax.nn.softmax(logits)
        log_pi = jax.nn.log_softmax(logits)

        q1, q2 = jax.vmap(q_apply, in_axes=(None, 0))(jax.lax.stop_gradient(q_params), states)
        min_q = jnp.minimum(q1, q2)

        # Policy loss: sum_a pi(a|s) * (alpha * log pi(a|s) - min_q(s,a))
        loss = jnp.mean(jnp.sum(pi * (jax.lax.stop_gradient(alpha) * log_pi - min_q), axis=1))
        entropy = -jnp.mean(jnp.sum(pi * log_pi, axis=1))
        return loss, entropy

    (a_loss, entropy), a_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_params)
    a_updates, actor_opt_state = actor_opt.update(a_grads, actor_opt_state, actor_params)
    actor_params = optax.apply_updates(actor_params, a_updates)

    # ── Alpha loss ──
    def alpha_loss_fn(la):
        return -jnp.exp(la) * jax.lax.stop_gradient(entropy + target_entropy)

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(log_alpha)
    alpha_update, alpha_opt_state = alpha_opt.update(alpha_grad, alpha_opt_state)
    log_alpha = optax.apply_updates(log_alpha, alpha_update)

    # ── Soft target update ──
    q_target_params = jax.tree.map(
        lambda p, tp: tau * p + (1.0 - tau) * tp,
        q_params, q_target_params,
    )

    return (actor_params, q_params, q_target_params, log_alpha,
            actor_opt_state, q_opt_state, alpha_opt_state,
            q_loss, a_loss, entropy, jnp.exp(log_alpha))


class ReplayBuffer:
    """Simple replay buffer for transitions (s, a, r, s', done)."""

    def __init__(self, capacity, n_features):
        self.capacity = capacity
        self.n_features = n_features
        self.states = np.zeros((capacity, n_features), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, n_features), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, states, actions, rewards, next_states, dones):
        n = states.shape[0]
        for i in range(n):
            self.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def sample(self, batch_size, key):
        idx = jax.random.randint(key, (batch_size,), 0, self.size)
        idx = np.array(idx)
        return (jnp.array(self.states[idx]),
                jnp.array(self.actions[idx]),
                jnp.array(self.rewards[idx]),
                jnp.array(self.next_states[idx]),
                jnp.array(self.dones[idx]))


@partial(jax.jit, static_argnames=(
    'n_episodes', 'n_actions', 'actor_apply', 'state_fn',
))
def collect_sac_transitions(
    n_episodes, key, actor_params, actor_apply,
    n_actions=51, state_fn=None,
):
    """Collect transitions for SAC replay buffer.

    Returns:
        states:      (n_episodes, 10, n_features)
        actions:     (n_episodes, 10)
        rewards:     (n_episodes, 10)
        next_states: (n_episodes, 10, n_features)
        dones:       (n_episodes, 10)
    """
    _state_fn = state_fn or raw_state_features

    def one_episode(key):
        def step(carry, t):
            key, b1, b2 = carry
            key, kv, kn, ka = jax.random.split(key, 4)

            v = jax.random.uniform(kv, minval=10.0, maxval=20.0)
            s_feat = _state_fn(v, jnp.float32(t), b1, b2)

            logits = actor_apply(actor_params, s_feat)
            action_idx = jax.random.categorical(ka, logits)

            cands = jnp.linspace(0.0, b1, n_actions)
            bid1 = cands[action_idx]
            bid2 = default_player_2(t, v, b2, b1, kn)

            reward = jnp.where(bid1 > bid2, v,
                     jnp.where(bid1 == bid2, v * 0.5, 0.0))

            new_b1 = b1 - bid1
            new_b2 = b2 - bid2
            done = jnp.float32(t == 9)

            # Next state (dummy for terminal)
            next_v = jax.random.uniform(kv, minval=10.0, maxval=20.0)  # won't matter for done=1
            next_s_feat = _state_fn(next_v, jnp.float32(t + 1), new_b1, new_b2)

            return (key, new_b1, new_b2), (s_feat, action_idx, reward, next_s_feat, done)

        _, outputs = jax.lax.scan(
            step,
            (key, jnp.float32(100.0), jnp.float32(100.0)),
            jnp.arange(10),
        )
        return outputs

    keys = jax.random.split(key, n_episodes)
    return jax.lax.map(one_episode, keys)


def run_sac_experiment(
    N=200_000,
    eval_every=1000,
    n_actions=51,
    lr_actor=3e-4,
    lr_q=3e-4,
    lr_alpha=3e-4,
    hidden=256,
    batch_size=256,
    buffer_size=100_000,
    n_eval=2000,
    seed=24,
    state_fn=None,
    n_state_features=4,
    target_entropy=None,
    tau=0.005,
    n_updates_per_chunk=10,
    warmup_episodes=5000,
):
    """Train a discrete SAC agent.

    Returns:
        dict with rewards, losses, params, best_reward
    """
    if target_entropy is None:
        target_entropy = -0.5 * np.log(n_actions)

    actor = PolicyNetwork(n_actions=n_actions, hidden=hidden)
    q_net = TwinQNetwork(n_actions=n_actions, hidden=hidden)

    key = jax.random.PRNGKey(seed)
    key, ak, qk = jax.random.split(key, 3)
    dummy = jnp.zeros(n_state_features)
    actor_params = actor.init(ak, dummy)
    q_params = q_net.init(qk, dummy)
    q_target_params = jax.tree.map(lambda x: x.copy(), q_params)
    log_alpha = jnp.array(0.0)

    actor_opt = optax.adam(lr_actor)
    q_opt = optax.adam(lr_q)
    alpha_opt = optax.adam(lr_alpha)

    actor_opt_state = actor_opt.init(actor_params)
    q_opt_state = q_opt.init(q_params)
    alpha_opt_state = alpha_opt.init(log_alpha)

    buffer = ReplayBuffer(buffer_size, n_state_features)

    n_chunks = N // eval_every
    best_actor_params = actor_params
    best_reward = -np.inf
    all_rewards, all_q_losses, all_actor_losses, all_entropy, all_alpha = [], [], [], [], []

    pbar = tqdm(range(n_chunks), desc='SAC')
    for chunk in pbar:
        key, gen_key, eval_key, train_key = jax.random.split(key, 4)

        # Collect transitions
        states, actions, rewards, next_states, dones = collect_sac_transitions(
            eval_every, gen_key, actor_params, actor.apply,
            n_actions=n_actions, state_fn=state_fn,
        )

        # Add to buffer
        s_np = np.array(states.reshape(-1, n_state_features))
        a_np = np.array(actions.reshape(-1))
        r_np = np.array(rewards.reshape(-1))
        ns_np = np.array(next_states.reshape(-1, n_state_features))
        d_np = np.array(dones.reshape(-1))
        buffer.add_batch(s_np, a_np, r_np, ns_np, d_np)

        if buffer.size < warmup_episodes * 10:
            all_rewards.append(0.0)
            all_q_losses.append(0.0)
            all_actor_losses.append(0.0)
            all_entropy.append(0.0)
            all_alpha.append(1.0)
            continue

        # Train
        chunk_q_loss, chunk_a_loss, chunk_ent, chunk_alpha = 0.0, 0.0, 0.0, 0.0
        for i in range(n_updates_per_chunk):
            train_key, sample_key = jax.random.split(train_key)
            s_b, a_b, r_b, ns_b, d_b = buffer.sample(batch_size, sample_key)

            (actor_params, q_params, q_target_params, log_alpha,
             actor_opt_state, q_opt_state, alpha_opt_state,
             q_loss, a_loss, ent, alpha_val) = sac_train_step(
                actor_params, q_params, q_target_params, log_alpha,
                actor_opt_state, q_opt_state, alpha_opt_state,
                actor.apply, q_net.apply,
                actor_opt, q_opt, alpha_opt,
                s_b, a_b, r_b, ns_b, d_b,
                gamma=1.0, target_entropy=target_entropy, tau=tau,
            )
            chunk_q_loss += float(q_loss)
            chunk_a_loss += float(a_loss)
            chunk_ent += float(ent)
            chunk_alpha += float(alpha_val)

        # Evaluate
        chunk_reward = float(evaluate_ppo(
            actor_params, eval_key, actor.apply,
            n_eval=n_eval, n_actions=n_actions, state_fn=state_fn,
        ))

        if chunk_reward > best_reward:
            best_reward = chunk_reward
            best_actor_params = jax.tree.map(lambda x: x.copy(), actor_params)

        all_rewards.append(chunk_reward)
        all_q_losses.append(chunk_q_loss / n_updates_per_chunk)
        all_actor_losses.append(chunk_a_loss / n_updates_per_chunk)
        all_entropy.append(chunk_ent / n_updates_per_chunk)
        all_alpha.append(chunk_alpha / n_updates_per_chunk)

        pbar.set_postfix(
            reward=f'{chunk_reward:.1f}', best=f'{best_reward:.1f}',
            alpha=f'{chunk_alpha / n_updates_per_chunk:.3f}',
        )

    return {
        'rewards': np.array(all_rewards),
        'q_losses': np.array(all_q_losses),
        'actor_losses': np.array(all_actor_losses),
        'entropy': np.array(all_entropy),
        'alpha': np.array(all_alpha),
        'actor_params_best': best_actor_params,
        'actor_params_final': actor_params,
        'best_reward': best_reward,
    }


# ── Tournament integration ──

def make_bid_fn_ppo(actor_params, state_fn, n_actions=51, hidden=256):
    """Create a JAX-native greedy bid function for a PPO/SAC agent."""
    actor = PolicyNetwork(n_actions=n_actions, hidden=hidden)
    _state_fn = state_fn or raw_state_features

    def bid_fn(v, t, own_b, opp_b):
        s_feat = _state_fn(jnp.asarray(v, dtype=jnp.float32),
                           jnp.asarray(t, dtype=jnp.float32),
                           jnp.asarray(own_b, dtype=jnp.float32),
                           jnp.asarray(opp_b, dtype=jnp.float32))
        logits = actor.apply(actor_params, s_feat)
        idx = jnp.where(own_b <= 1e-10, jnp.int32(0), jnp.argmax(logits))
        cands = jnp.linspace(0.0, own_b, n_actions)
        return cands[idx]

    return bid_fn


def export_policy_weights(actor_params, filepath='ppo_sac_weights.npz', feature_type='tile'):
    """Export actor weights for the standalone rlagent.

    Args:
        actor_params: Flax parameter pytree
        filepath: output .npz path
        feature_type: 'raw' (4-dim), 'binned' (41-dim), or 'tile' (512-dim)
    """
    _p = actor_params['params']
    # Encode feature_type as int: 0=raw, 1=binned, 2=tile
    ft_map = {'raw': 0, 'binned': 1, 'tile': 2}
    np.savez(filepath,
             W1=np.array(_p['Dense_0']['kernel']),
             B1=np.array(_p['Dense_0']['bias']),
             W2=np.array(_p['Dense_1']['kernel']),
             B2=np.array(_p['Dense_1']['bias']),
             W3=np.array(_p['Dense_2']['kernel']),
             B3=np.array(_p['Dense_2']['bias']),
             feature_type=np.array(ft_map.get(feature_type, 0)))
