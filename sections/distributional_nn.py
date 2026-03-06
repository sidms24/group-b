"""Distributional RL (C51) with neural network for the bidding game.

Replace linear function approximation with a small MLP that learns
its own representation from raw state-action inputs. Use categorical
distributional RL (Bellemare et al. 2017) to model the full return
distribution rather than just the expected value.

Architecture:  n_inputs raw features -> 128 -> 128 -> N_ATOMS logits
Training:      Batch MC returns -> categorical projection -> CE loss -> Adam
"""

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
from functools import partial
from tqdm import tqdm

from sections.policies import default_player_2

# Distributional support
N_ATOMS = 51
R_MIN = 0.0
R_MAX = 150.0  # max possible return: 10 rounds x 15 avg prize
ATOMS = jnp.linspace(R_MIN, R_MAX, N_ATOMS)


# Network

class C51Network(nn.Module):
    """Small MLP outputting a categorical return distribution."""
    n_atoms: int = N_ATOMS
    hidden: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_atoms)(x)
        return x  # raw logits (n_atoms,)


# Raw feature extraction

def raw_features(v, t, own_b, opp_b, candidates):
    """Normalised raw inputs, no hand-crafted bins. """
    n = candidates.shape[0]
    safe_b = jnp.where(own_b > 1e-10, own_b, 1.0)
    return jnp.stack([
        jnp.broadcast_to((v - 10.0) / 10.0, (n,)),
        jnp.broadcast_to(t / 9.0, (n,)),
        jnp.broadcast_to(own_b / 100.0, (n,)),
        jnp.broadcast_to(opp_b / 100.0, (n,)),
        candidates / safe_b,
    ], axis=1)


# Categorical projection

def project_return(G, atoms=ATOMS, r_min=R_MIN, r_max=R_MAX, n_atoms=N_ATOMS):
    """Project a scalar return onto the categorical atom support."""
    G_clipped = jnp.clip(G, r_min, r_max)
    spacing = (r_max - r_min) / (n_atoms - 1)
    b = (G_clipped - r_min) / spacing
    lower = jnp.floor(b).astype(jnp.int32)
    upper = jnp.clip(lower + 1, 0, n_atoms - 1)
    lower = jnp.clip(lower, 0, n_atoms - 1)
    target = jnp.zeros(n_atoms)
    target = target.at[lower].add(upper - b)
    target = target.at[upper].add(b - lower)
    return target


# Episode generation with NN

@partial(jax.jit, static_argnames=('n_episodes', 'n_actions', 'net_apply', 'feature_fn'))
def generate_episodes_nn(n_episodes, key, epsilon, params, net_apply,
                         n_actions=51, feature_fn=None):
    """Generate episodes using the C51 network for action selection. """
    _feature_fn = feature_fn or raw_features

    def one_episode(key):
        def step(carry, t):
            key, b1, b2 = carry
            key, kv, kn, ke1, ke2 = jax.random.split(key, 5)

            v = jax.random.uniform(kv, minval=10.0, maxval=20.0)
            cands = jnp.linspace(0.0, b1, n_actions)

            raw = _feature_fn(v, jnp.float32(t), b1, b2, cands)

            # NN forward pass -> expected Q-values
            logits = jax.vmap(net_apply, in_axes=(None, 0))(params, raw)
            probs = jax.nn.softmax(logits)
            q_vals = jnp.sum(probs * ATOMS[None, :], axis=1)

            # Epsilon-greedy action selection
            greedy = jnp.argmax(q_vals)
            rand = jax.random.randint(ke1, (), 0, n_actions)
            idx = jnp.where(jax.random.uniform(ke2) < epsilon, rand, greedy)
            bid1 = cands[idx]

            # Opponent
            bid2 = default_player_2(t, v, b2, b1, kn)

            base_reward = jnp.where(bid1 > bid2, v,
                          jnp.where(bid1 == bid2, v * 0.5, 0.0))

            chosen_raw = raw[idx]
            return (key, b1 - bid1, b2 - bid2), (chosen_raw, base_reward)

        _, (raw_phis, rewards) = jax.lax.scan(
            step,
            (key, jnp.float32(100.0), jnp.float32(100.0)),
            jnp.arange(10),
        )
        return raw_phis, rewards

    keys = jax.random.split(key, n_episodes)
    return jax.lax.map(one_episode, keys)


# Episode generation with self-play

@partial(jax.jit, static_argnames=('n_episodes', 'n_actions', 'net_apply', 'feature_fn'))
def generate_episodes_nn_selfplay(
    n_episodes, key, epsilon, params, opponent_params, net_apply,
    n_actions=51, feature_fn=None,
):
    """Generate episodes with agent vs frozen opponent network."""
    _feature_fn = feature_fn or raw_features

    def one_episode(key):
        def step(carry, t):
            key, b1, b2 = carry
            key, kv, ke1, ke2 = jax.random.split(key, 4)

            v = jax.random.uniform(kv, minval=10.0, maxval=20.0)

            # Player 1 (agent): epsilon-greedy
            cands_p1 = jnp.linspace(0.0, b1, n_actions)
            raw = _feature_fn(v, jnp.float32(t), b1, b2, cands_p1)

            logits_p1 = jax.vmap(net_apply, in_axes=(None, 0))(params, raw)
            probs_p1 = jax.nn.softmax(logits_p1)
            q_p1 = jnp.sum(probs_p1 * ATOMS[None, :], axis=1)

            greedy = jnp.argmax(q_p1)
            rand = jax.random.randint(ke1, (), 0, n_actions)
            idx = jnp.where(jax.random.uniform(ke2) < epsilon, rand, greedy)
            bid1 = cands_p1[idx]

            # Player 2 (opponent): greedy from opponent_params
            cands_p2 = jnp.linspace(0.0, b2, n_actions)
            raw_opp = _feature_fn(v, jnp.float32(t), b2, b1, cands_p2)

            logits_p2 = jax.vmap(net_apply, in_axes=(None, 0))(opponent_params, raw_opp)
            probs_p2 = jax.nn.softmax(logits_p2)
            q_p2 = jnp.sum(probs_p2 * ATOMS[None, :], axis=1)

            opp_idx = jnp.where(b2 <= 1e-10, jnp.int32(0), jnp.argmax(q_p2))
            bid2 = cands_p2[opp_idx]

            base_reward = jnp.where(bid1 > bid2, v,
                          jnp.where(bid1 == bid2, v * 0.5, 0.0))

            chosen_raw = raw[idx]
            return (key, b1 - bid1, b2 - bid2), (chosen_raw, base_reward)

        _, (raw_phis, rewards) = jax.lax.scan(
            step,
            (key, jnp.float32(100.0), jnp.float32(100.0)),
            jnp.arange(10),
        )
        return raw_phis, rewards

    keys = jax.random.split(key, n_episodes)
    return jax.lax.map(one_episode, keys)


# Training step

@partial(jax.jit, static_argnames=('net_apply', 'optimizer'))
def train_step(params, opt_state, net_apply, optimizer,
               phis_flat, targets_flat):
    """One gradient step on a mini-batch. """
    def loss_fn(params):
        logits = jax.vmap(net_apply, in_axes=(None, 0))(params, phis_flat)
        log_probs = jax.nn.log_softmax(logits)
        return -jnp.mean(jnp.sum(targets_flat * log_probs, axis=1))

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# Greedy evaluation

@partial(jax.jit, static_argnames=('n_eval', 'n_actions', 'net_apply', 'feature_fn'))
def evaluate_nn(params, key, net_apply, n_eval=2000, n_actions=51, feature_fn=None):
    """Evaluate the C51 agent greedily against default_player_2."""
    _feature_fn = feature_fn or raw_features

    def one_eval(key):
        def step(carry, t):
            key, b1, b2, total = carry
            key, kv, kn = jax.random.split(key, 3)

            v = jax.random.uniform(kv, minval=10.0, maxval=20.0)
            cands = jnp.linspace(0.0, b1, n_actions)
            raw = _feature_fn(v, jnp.float32(t), b1, b2, cands)

            logits = jax.vmap(net_apply, in_axes=(None, 0))(params, raw)
            probs = jax.nn.softmax(logits)
            q_vals = jnp.sum(probs * ATOMS[None, :], axis=1)

            # Greedy (bid 0 if budget exhausted)
            idx = jnp.where(b1 <= 1e-10, jnp.int32(0), jnp.argmax(q_vals))
            bid1 = cands[idx]
            bid2 = default_player_2(t, v, b2, b1, kn)

            reward = jnp.where(bid1 > bid2, v,
                     jnp.where(bid1 == bid2, v * 0.5, 0.0))
            return (key, b1 - bid1, b2 - bid2, total + reward), None

        (_, _, _, total), _ = jax.lax.scan(
            step,
            (key, jnp.float32(100.0), jnp.float32(100.0), 0.0),
            jnp.arange(10),
        )
        return total

    keys = jax.random.split(key, n_eval)
    return jnp.mean(jax.vmap(one_eval)(keys))


# Full control experiment

def run_c51_experiment(
    N=200_000,
    eval_every=1000,
    n_actions=51,
    lr=3e-4,
    n_grad_steps=10,
    batch_size=256,
    hidden=128,
    n_eval=2000,
    seed=24,
    eps_fn=None,
    feature_fn=None,
    n_inputs=5,
):
    """Train a C51 distributional agent with MC returns and Adam.

    Args:
        N: total episodes
        eval_every: episodes per chunk
        n_actions: bid discretisation
        lr: Adam learning rate
        n_grad_steps: mini-batch gradient steps per chunk
        batch_size: mini-batch size
        hidden: network hidden layer width
        n_eval: evaluation episodes per checkpoint
        seed: random seed
        eps_fn: epsilon schedule callable(ep) -> float; defaults to cosine 1->0.01
        feature_fn: feature function (v, t, b1, b2, candidates) -> (n_actions, n_inputs).
                    Defaults to raw_features (5-dim).
        n_inputs: input dimensionality matching feature_fn output. Default 5.

    Returns:
        dict with rewards, losses, params_best, best_reward
    """
    if eps_fn is None:
        def eps_fn(ep):
            return 0.01 + 0.5 * (1.0 - 0.01) * (1 + np.cos(np.pi * ep / N))

    # Initialise network and optimizer
    net = C51Network(n_atoms=N_ATOMS, hidden=hidden)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = net.init(init_key, jnp.zeros(n_inputs))

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    n_chunks = N // eval_every
    eps_arr = np.array([eps_fn(int(ep)) for ep in range(N)], dtype=np.float32)
    chunk_eps = eps_arr[::eval_every][:n_chunks]

    best_params = params
    best_reward = -np.inf
    all_rewards, all_losses = [], []

    pbar = tqdm(range(n_chunks), desc='C51')
    for chunk in pbar:
        key, gen_key, eval_key, shuf_key = jax.random.split(key, 4)
        epsilon = float(chunk_eps[chunk])

        # Collect episodes (jax.lax.map: sequential, constant memory)
        raw_phis, rewards = generate_episodes_nn(
            eval_every, gen_key, epsilon, params, net.apply,
            n_actions=n_actions, feature_fn=feature_fn,
        )

        # Compute MC returns (backward cumsum, gamma=1)
        returns = jnp.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]

        # Project returns onto atoms
        targets = jax.vmap(jax.vmap(project_return))(returns)

        # Flatten for mini-batch training
        phis_flat = raw_phis.reshape(-1, n_inputs)
        tgts_flat = targets.reshape(-1, N_ATOMS)
        n_samples = phis_flat.shape[0]

        # Mini-batch gradient steps
        chunk_loss = 0.0
        for step_i in range(n_grad_steps):
            shuf_key, perm_key = jax.random.split(shuf_key)
            perm = jax.random.permutation(perm_key, n_samples)
            mb_idx = perm[:batch_size]
            params, opt_state, loss = train_step(
                params, opt_state, net.apply, optimizer,
                phis_flat[mb_idx], tgts_flat[mb_idx],
            )
            chunk_loss += float(loss)

        # Evaluate
        chunk_reward = float(evaluate_nn(
            params, eval_key, net.apply,
            n_eval=n_eval, n_actions=n_actions, feature_fn=feature_fn,
        ))

        if chunk_reward > best_reward:
            best_reward = chunk_reward
            best_params = jax.tree.map(lambda x: x.copy(), params)

        all_rewards.append(chunk_reward)
        all_losses.append(chunk_loss / n_grad_steps)

        pbar.set_postfix(eps=f'{epsilon:.3f}', reward=f'{chunk_reward:.1f}',
                         best=f'{best_reward:.1f}')

    return {
        'rewards': np.array(all_rewards),
        'losses': np.array(all_losses),
        'params_best': best_params,
        'params_final': params,
        'best_reward': best_reward,
    }


# Self-play control experiment

def run_c51_selfplay_experiment(
    N=200_000,
    eval_every=1000,
    n_actions=51,
    lr=3e-4,
    n_grad_steps=10,
    batch_size=256,
    hidden=128,
    n_eval=2000,
    seed=24,
    eps_fn=None,
    feature_fn=None,
    n_inputs=5,
    opponent_update_every=10,
):
    """Train a C51 agent via self-play against a frozen opponent network. """
    if eps_fn is None:
        def eps_fn(ep):
            return 0.01 + 0.5 * (1.0 - 0.01) * (1 + np.cos(np.pi * ep / N))

    net = C51Network(n_atoms=N_ATOMS, hidden=hidden)
    key = jax.random.PRNGKey(seed)
    key, init_key, opp_init_key = jax.random.split(key, 3)
    params = net.init(init_key, jnp.zeros(n_inputs))
    opponent_params = net.init(opp_init_key, jnp.zeros(n_inputs))

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    n_chunks = N // eval_every
    eps_arr = np.array([eps_fn(int(ep)) for ep in range(N)], dtype=np.float32)
    chunk_eps = eps_arr[::eval_every][:n_chunks]

    best_params = params
    best_reward = -np.inf
    all_rewards, all_losses, selfplay_rewards = [], [], []

    pbar = tqdm(range(n_chunks), desc='C51 self-play')
    for chunk in pbar:
        key, gen_key, eval_key, shuf_key = jax.random.split(key, 4)
        epsilon = float(chunk_eps[chunk])

        # Update opponent periodically
        if chunk > 0 and chunk % opponent_update_every == 0:
            opponent_params = jax.tree.map(lambda x: x.copy(), params)

        # Collect episodes via self-play (jax.lax.map: sequential, constant memory)
        raw_phis, rewards = generate_episodes_nn_selfplay(
            eval_every, gen_key, epsilon, params, opponent_params, net.apply,
            n_actions=n_actions, feature_fn=feature_fn,
        )

        sp_reward = float(jnp.mean(jnp.sum(rewards, axis=1)))
        selfplay_rewards.append(sp_reward)

        # MC returns (gamma=1)
        returns = jnp.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
        targets = jax.vmap(jax.vmap(project_return))(returns)

        phis_flat = raw_phis.reshape(-1, n_inputs)
        tgts_flat = targets.reshape(-1, N_ATOMS)
        n_samples = phis_flat.shape[0]

        # Mini-batch gradient steps
        chunk_loss = 0.0
        for step_i in range(n_grad_steps):
            shuf_key, perm_key = jax.random.split(shuf_key)
            perm = jax.random.permutation(perm_key, n_samples)
            mb_idx = perm[:batch_size]
            params, opt_state, loss = train_step(
                params, opt_state, net.apply, optimizer,
                phis_flat[mb_idx], tgts_flat[mb_idx],
            )
            chunk_loss += float(loss)

        # Evaluate against fixed heuristic
        chunk_reward = float(evaluate_nn(
            params, eval_key, net.apply,
            n_eval=n_eval, n_actions=n_actions, feature_fn=feature_fn,
        ))

        if chunk_reward > best_reward:
            best_reward = chunk_reward
            best_params = jax.tree.map(lambda x: x.copy(), params)

        all_rewards.append(chunk_reward)
        all_losses.append(chunk_loss / n_grad_steps)

        pbar.set_postfix(eps=f'{epsilon:.3f}', reward=f'{chunk_reward:.1f}',
                         best=f'{best_reward:.1f}', sp_r=f'{sp_reward:.1f}')

    return {
        'rewards': np.array(all_rewards),
        'selfplay_rewards': np.array(selfplay_rewards),
        'losses': np.array(all_losses),
        'params_best': best_params,
        'params_final': params,
        'best_reward': best_reward,
    }


# ── Tournament utilities ──

def make_bid_fn_c51(params, feature_fn, n_actions=51, hidden=256):
    """Create a JAX-native greedy bid function for a C51 agent."""
    net = C51Network(n_atoms=N_ATOMS, hidden=hidden)

    def bid_fn(v, t, own_b, opp_b):
        cands = jnp.linspace(0.0, own_b, n_actions)
        raw = feature_fn(jnp.asarray(v, dtype=jnp.float32),
                         jnp.asarray(t, dtype=jnp.float32),
                         jnp.asarray(own_b, dtype=jnp.float32),
                         jnp.asarray(opp_b, dtype=jnp.float32), cands)
        logits = jax.vmap(net.apply, in_axes=(None, 0))(params, raw)
        probs = jax.nn.softmax(logits)
        q_vals = jnp.sum(probs * ATOMS[None, :], axis=1)
        idx = jnp.where(own_b <= 1e-10, jnp.int32(0), jnp.argmax(q_vals))
        return cands[idx]

    return bid_fn


def make_bid_fn_linear(theta, feature_fn, n_actions=51):
    """Create a JAX-native greedy bid function for a linear FA agent."""
    theta_jnp = jnp.array(theta)

    def bid_fn(v, t, own_b, opp_b):
        cands = jnp.linspace(0.0, own_b, n_actions)
        phis = feature_fn(jnp.asarray(v, dtype=jnp.float32),
                          jnp.asarray(t, dtype=jnp.float32),
                          jnp.asarray(own_b, dtype=jnp.float32),
                          jnp.asarray(opp_b, dtype=jnp.float32), cands)
        q_vals = phis @ theta_jnp
        idx = jnp.where(own_b <= 1e-10, jnp.int32(0), jnp.argmax(q_vals))
        return cands[idx]

    return bid_fn


def make_bid_fn_heuristic():
    """Create a JAX-native bid function for the default heuristic opponent."""
    def bid_fn(v, t, own_b, opp_b, key):
        return default_player_2(t, v, own_b, opp_b, key)

    return bid_fn


def play_head_to_head(bid_fn_a, bid_fn_b, n_games=1000, seed=42,
                      a_is_heuristic=False, b_is_heuristic=False):
    """JIT-compiled head-to-head: vmap over games, lax.scan over rounds. """
    def one_game(key):
        def one_round(carry, t):
            key, ba, bb, ra, rb = carry
            key, kv, ka, kb = jax.random.split(key, 4)

            v = jax.random.uniform(kv, minval=10.0, maxval=20.0)

            if a_is_heuristic:
                bid_a = jnp.clip(bid_fn_a(v, t, ba, bb, ka), 0.0, ba)
            else:
                bid_a = jnp.clip(bid_fn_a(v, t, ba, bb), 0.0, ba)

            if b_is_heuristic:
                bid_b = jnp.clip(bid_fn_b(v, t, bb, ba, kb), 0.0, bb)
            else:
                bid_b = jnp.clip(bid_fn_b(v, t, bb, ba), 0.0, bb)

            reward_a = jnp.where(bid_a > bid_b, v,
                       jnp.where(bid_a == bid_b, v / 2, 0.0))
            reward_b = jnp.where(bid_b > bid_a, v,
                       jnp.where(bid_a == bid_b, v / 2, 0.0))

            return (key, ba - bid_a, bb - bid_b, ra + reward_a, rb + reward_b), None

        init = (key, jnp.float32(100.0), jnp.float32(100.0),
                jnp.float32(0.0), jnp.float32(0.0))
        (_, _, _, total_a, total_b), _ = jax.lax.scan(one_round, init, jnp.arange(10))
        return total_a, total_b

    keys = jax.random.split(jax.random.PRNGKey(seed), n_games)
    rewards_a, rewards_b = jax.vmap(one_game)(keys)

    wins_a = jnp.sum(rewards_a > rewards_b)
    wins_b = jnp.sum(rewards_b > rewards_a)

    return {
        'mean_a': float(jnp.mean(rewards_a)),
        'mean_b': float(jnp.mean(rewards_b)),
        'win_rate_a': float(wins_a / n_games),
        'win_rate_b': float(wins_b / n_games),
    }


def round_robin_tournament(agents, n_games=1000, seed=42):
    """Run a full round-robin tournament between all agents. """
    names = list(agents.keys())
    n = len(names)
    reward_matrix = np.zeros((n, n))
    win_matrix = np.zeros((n, n))

    rng = np.random.RandomState(seed)
    total_matchups = n * (n - 1) // 2
    matchup = 0

    for i in range(n):
        for j in range(i + 1, n):
            matchup += 1
            s1 = rng.randint(0, 2**31)
            s2 = rng.randint(0, 2**31)

            a_heur = names[i] == 'Heuristic'
            b_heur = names[j] == 'Heuristic'

            # Direction 1: i as P1, j as P2
            r1 = play_head_to_head(agents[names[i]], agents[names[j]],
                                   n_games=n_games, seed=s1,
                                   a_is_heuristic=a_heur, b_is_heuristic=b_heur)
            # Direction 2: j as P1, i as P2
            r2 = play_head_to_head(agents[names[j]], agents[names[i]],
                                   n_games=n_games, seed=s2,
                                   a_is_heuristic=b_heur, b_is_heuristic=a_heur)

            # Average across both seating positions
            win_matrix[i, j] = (r1['win_rate_a'] + r2['win_rate_b']) / 2
            win_matrix[j, i] = (r1['win_rate_b'] + r2['win_rate_a']) / 2
            reward_matrix[i, j] = (r1['mean_a'] + r2['mean_b']) / 2
            reward_matrix[j, i] = (r1['mean_b'] + r2['mean_a']) / 2

            print(f'  [{matchup}/{total_matchups}] {names[i]} vs {names[j]}: '
                  f'{win_matrix[i,j]:.3f} - {win_matrix[j,i]:.3f}')

    # Mean win rate and reward against all opponents
    total_wins = np.array([
        np.mean([win_matrix[i, j] for j in range(n) if j != i])
        for i in range(n)
    ])
    total_reward = np.array([
        np.mean([reward_matrix[i, j] for j in range(n) if j != i])
        for i in range(n)
    ])

    ranking = sorted(zip(names, total_wins), key=lambda x: -x[1])

    return {
        'names': names,
        'win_matrix': win_matrix,
        'reward_matrix': reward_matrix,
        'total_wins': total_wins,
        'total_reward': total_reward,
        'ranking': ranking,
    }
