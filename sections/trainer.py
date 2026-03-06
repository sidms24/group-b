"""Training pipeline.

run_mc_experiment  - train only the MC agent (prediction or control)
run_td_experiment  - train only the TD agent (prediction or control)
run_experiment_online - run both agents and produce combined plots/results
                        (same return format as the original combined trainer)

Each individual trainer returns
    results : dict  {config_name: {'thetas', 'losses', 'alphas', 'theta_final'}}
    empirical: float
        prediction mode - mean total reward per episode under the fixed policy
        control mode    - mean greedy-policy reward of the final trained theta

run_experiment_online returns the original combined format
    results : dict  {config_name: {'mc_eval', 'td_eval', 'mc_losses', 'td_losses',
                                   'alphas', 'mc_theta_final', 'td_theta_final',
                                   'mc_thetas', 'td_thetas'}}
    empirical_return : float (prediction) or {'mc': float, 'td': float} (control)
"""

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

from sections.features import N_FEATURES, compute_features_batch_jax
from sections.policies import default_player_2, uniform_policy
from sections.episodes import generate_episodes, mse, N_ACTIONS
from sections.mc import MC
from sections.td import TD
from sections.schedules import epsilon_schedule_exp
from sections.evaluation import evaluate_theta
from sections.visualisations import (
    mc_td_pred_eval, mc_td_control_eval, _plot_summary, plot_standalone_results
)


SEED = 24


# Helper functions shared by all experiment runners.

def _zeros(n_features=N_FEATURES):
    """Create zero-initialised theta vector."""
    return jnp.zeros(n_features)


def _build_eps(N, epsilon, epsilon_scheduler):
    """Build episode index array and epsilon schedule array."""
    episodes_np = np.arange(N)
    if epsilon_scheduler is None:
        eps_arr = np.array([epsilon_schedule_exp(ep) for ep in episodes_np],
                           dtype=np.float32)
    else:
        eps_arr = np.array([epsilon_scheduler(ep) for ep in episodes_np],
                           dtype=np.float32)
    return episodes_np, eps_arr


# MC experiment runner for both prediction and control modes.

def run_mc_experiment(configs_fn, prediction=True, opponent_fn=None, loss_fn=mse,
                      epsilon=0.0, epsilon_scheduler=None, eval_every=1000,
                      N=100_000, reg=False, plot=False,
                      n_actions=N_ACTIONS, n_features=N_FEATURES,
                      feature_fn=compute_features_batch_jax, zero_sum=False,
                      lambda_l1=0.0, lambda_l2=0.0, n_eval=500):
    """Train the MC agent only.

    Parameters
    ----------
    n_actions : int
        Number of discrete bid candidates per step.
    n_features : int
        Dimensionality of the feature vector returned by feature_fn.
    feature_fn : callable
        feature_fn(v, t, b1, b2, candidates) -> (n_actions, n_features).

    Returns
    -------
    results : dict
        {config_name: {'thetas', 'losses', 'alphas', 'theta_final'}}
    empirical : float
        prediction: mean total reward under the fixed policy
        control: greedy-policy reward of the final theta
    """
    key = jax.random.PRNGKey(SEED)
    opp = opponent_fn or (uniform_policy if prediction else default_player_2)
    theta_init = _zeros(n_features)
    episodes_np, eps_arr = _build_eps(N, epsilon, epsilon_scheduler)
    configs = configs_fn(episodes_np, eps_arr)
    results = {}

    if prediction:
        # Generate all episodes upfront under fixed policies for offline training.
        phis, rewards, _, _, _ = generate_episodes(
            N, key, epsilon=epsilon, theta=theta_init,
            opponent_fn=opp, prediction=True,
            n_actions=n_actions, feature_fn=feature_fn, zero_sum=zero_sum,
        )
        phis.block_until_ready()
        empirical = float(rewards.sum(axis=1).mean())

        for name, alpha_fn in tqdm(configs.items(), desc="MC"):
            alphas_jax = jnp.array(
                np.array([alpha_fn(ep) for ep in episodes_np], dtype=np.float32)
            )
            mc_thetas, mc_losses = MC(
                phis, rewards, alphas_jax,
                eval_every=eval_every, loss_fn=loss_fn, reg=reg,
                lambda_l1=lambda_l1, lambda_l2=lambda_l2,
            )
            results[name] = {
                'thetas': np.array(mc_thetas),
                'losses': np.array(mc_losses),
                'alphas': np.array([alpha_fn(ep) for ep in episodes_np],
                                   dtype=np.float32),
                'theta_final': np.array(mc_thetas[-1]),
            }

    else:
        # Control mode: collect on-policy data in chunks with warm-start.
        n_chunks = N // eval_every
        mc_keys = jax.random.split(jax.random.PRNGKey(SEED), n_chunks)
        mc_eval_keys = jax.random.split(jax.random.PRNGKey(SEED + 10_000), n_chunks)
        chunk_eps_arr = np.array(eps_arr[::eval_every][:n_chunks], dtype=np.float32)

        for name, alpha_fn in tqdm(configs.items(), desc="MC"):
            alphas_np = np.array([alpha_fn(ep) for ep in episodes_np], dtype=np.float32)
            alphas_chunked = jnp.array(alphas_np).reshape(n_chunks, eval_every)

            warm_theta = theta_init
            best_theta = theta_init
            best_reward = -np.inf
            all_thetas, all_losses = [], []

            for chunk in range(n_chunks):
                chunk_eps = float(chunk_eps_arr[chunk])

                phis, rewards, _, _, _ = generate_episodes(
                    eval_every, mc_keys[chunk],
                    epsilon=chunk_eps, theta=warm_theta,
                    opponent_fn=opp, prediction=False,
                    n_actions=n_actions, feature_fn=feature_fn, zero_sum=zero_sum,
                )
                phis.block_until_ready()

                thetas_chunk, loss_chunk = MC(
                    phis, rewards,
                    alphas_chunked[chunk],
                    eval_every=eval_every, loss_fn=loss_fn, reg=reg,
                    theta_init=warm_theta,
                    lambda_l1=lambda_l1, lambda_l2=lambda_l2,
                )
                trained_theta = thetas_chunk[-1]

                chunk_reward = float(evaluate_theta(
                    jnp.array(trained_theta), mc_eval_keys[chunk], n_eval=n_eval,
                    n_actions=n_actions, feature_fn=feature_fn,
                ))
                if chunk_reward > best_reward:
                    best_reward = chunk_reward
                    best_theta = trained_theta

                warm_theta = trained_theta

                all_thetas.append(np.array(trained_theta))
                all_losses.append(float(loss_chunk[-1]))

            results[name] = {
                'thetas': np.array(all_thetas),
                'losses': np.array(all_losses),
                'alphas': alphas_np,
                'theta_final': all_thetas[-1],
                'theta_best': np.array(best_theta),
            }

        last_theta = list(results.values())[-1]['theta_best']
        empirical = float(evaluate_theta(
            jnp.array(last_theta), jax.random.PRNGKey(999), n_eval=500,
            n_actions=n_actions, feature_fn=feature_fn,
        ))

    if plot:
        plot_standalone_results(results, 'MC', eval_every, prediction,
                                empirical=empirical, n_actions=n_actions,
                                feature_fn=feature_fn, n_eval=n_eval)

    return results, empirical


# TD experiment runner for both prediction and control modes.

def run_td_experiment(configs_fn, prediction=True, opponent_fn=None, loss_fn=mse,
                      epsilon=0.0, epsilon_scheduler=None, eval_every=1000,
                      N=100_000, reg=False, plot=False,
                      n_actions=N_ACTIONS, n_features=N_FEATURES,
                      feature_fn=compute_features_batch_jax, zero_sum=False,
                      lambda_l1=0.0, lambda_l2=0.0, n_eval=500):
    """Train the TD agent only. """
    key = jax.random.PRNGKey(SEED)
    opp = opponent_fn or (uniform_policy if prediction else default_player_2)
    theta_init = _zeros(n_features)
    episodes_np, eps_arr = _build_eps(N, epsilon, epsilon_scheduler)
    configs = configs_fn(episodes_np, eps_arr)
    results = {}

    if prediction:
        phis, rewards, _, _, _ = generate_episodes(
            N, key, epsilon=epsilon, theta=theta_init,
            opponent_fn=opp, prediction=True,
            n_actions=n_actions, feature_fn=feature_fn, zero_sum=zero_sum,
        )
        phis.block_until_ready()
        empirical = float(rewards.sum(axis=1).mean())

        for name, alpha_fn in tqdm(configs.items(), desc="TD"):
            alphas_jax = jnp.array(
                np.array([alpha_fn(ep) for ep in episodes_np], dtype=np.float32)
            )
            td_thetas, td_losses = TD(
                phis, rewards, alphas_jax,
                eval_every=eval_every, loss_fn=loss_fn, reg=reg,
                lambda_l1=lambda_l1, lambda_l2=lambda_l2,
            )
            results[name] = {
                'thetas': np.array(td_thetas),
                'losses': np.array(td_losses),
                'alphas': np.array([alpha_fn(ep) for ep in episodes_np],
                                   dtype=np.float32),
                'theta_final': np.array(td_thetas[-1]),
            }

    else:
        n_chunks = N // eval_every
        td_keys = jax.random.split(jax.random.PRNGKey(SEED + n_chunks), n_chunks)
        td_eval_keys = jax.random.split(jax.random.PRNGKey(SEED + 20_000), n_chunks)
        chunk_eps_arr = np.array(eps_arr[::eval_every][:n_chunks], dtype=np.float32)

        for name, alpha_fn in tqdm(configs.items(), desc="TD"):
            alphas_np = np.array([alpha_fn(ep) for ep in episodes_np], dtype=np.float32)
            alphas_chunked = jnp.array(alphas_np).reshape(n_chunks, eval_every)

            warm_theta = theta_init
            best_theta = theta_init
            best_reward = -np.inf
            all_thetas, all_losses = [], []

            for chunk in range(n_chunks):
                chunk_eps = float(chunk_eps_arr[chunk])

                phis, rewards, _, _, _ = generate_episodes(
                    eval_every, td_keys[chunk],
                    epsilon=chunk_eps, theta=warm_theta,
                    opponent_fn=opp, prediction=False,
                    n_actions=n_actions, feature_fn=feature_fn, zero_sum=zero_sum,
                )
                phis.block_until_ready()

                thetas_chunk, loss_chunk = TD(
                    phis, rewards,
                    alphas_chunked[chunk],
                    eval_every=eval_every, loss_fn=loss_fn, reg=reg,
                    theta_init=warm_theta,
                    lambda_l1=lambda_l1, lambda_l2=lambda_l2,
                )
                trained_theta = thetas_chunk[-1]

                chunk_reward = float(evaluate_theta(
                    jnp.array(trained_theta), td_eval_keys[chunk], n_eval=n_eval,
                    n_actions=n_actions, feature_fn=feature_fn,
                ))
                if chunk_reward > best_reward:
                    best_reward = chunk_reward
                    best_theta = trained_theta

                warm_theta = trained_theta

                all_thetas.append(np.array(trained_theta))
                all_losses.append(float(loss_chunk[-1]))

            results[name] = {
                'thetas': np.array(all_thetas),
                'losses': np.array(all_losses),
                'alphas': alphas_np,
                'theta_final': all_thetas[-1],
                'theta_best': np.array(best_theta),
            }

        last_theta = list(results.values())[-1]['theta_best']
        empirical = float(evaluate_theta(
            jnp.array(last_theta), jax.random.PRNGKey(998), n_eval=500,
            n_actions=n_actions, feature_fn=feature_fn,
        ))

    if plot:
        plot_standalone_results(results, 'TD', eval_every, prediction,
                                empirical=empirical, n_actions=n_actions,
                                feature_fn=feature_fn, n_eval=n_eval)

    return results, empirical


# Combined experiment runner that trains both MC and TD and produces shared plots.

def run_experiment_online(configs_fn, prediction=True, opponent_fn=None, loss_fn=mse,
                          epsilon=0.0, epsilon_scheduler=None, eval_every=1000,
                          N=100_000, plot=True, reg=False,
                          n_actions=N_ACTIONS, n_features=N_FEATURES,
                          feature_fn=compute_features_batch_jax, zero_sum=False,
                          lambda_l1=0.0, lambda_l2=0.0, n_eval=500):
    """Run both MC and TD experiments and produce combined plots/results.

    Returns
    -------
    results : dict
        {config_name: {'mc_eval', 'td_eval', 'mc_losses', 'td_losses',
                       'alphas', 'mc_theta_final', 'td_theta_final',
                       'mc_thetas', 'td_thetas'}}
    empirical_return : float (prediction) or {'mc': float, 'td': float} (control)
    """
    shared_kwargs = dict(
        prediction=prediction, opponent_fn=opponent_fn, loss_fn=loss_fn,
        epsilon=epsilon, epsilon_scheduler=epsilon_scheduler,
        eval_every=eval_every, N=N, reg=reg,
        n_actions=n_actions, n_features=n_features, feature_fn=feature_fn,
        zero_sum=zero_sum, lambda_l1=lambda_l1, lambda_l2=lambda_l2, n_eval=n_eval,
    )

    mc_results, mc_empirical = run_mc_experiment(configs_fn, **shared_kwargs)
    td_results, td_empirical = run_td_experiment(configs_fn, **shared_kwargs)

    eval_fn = mc_td_pred_eval if prediction else mc_td_control_eval
    episodes_np, eps_arr = _build_eps(N, epsilon, epsilon_scheduler)

    combined = {}
    for name in mc_results:
        mc = mc_results[name]
        td = td_results[name]

        mc_eval, td_eval = eval_fn(
            mc['thetas'], td['thetas'],
            mc['losses'], td['losses'],
            plot=plot, title_suffix=f'({name})',
            eval_every=eval_every,
            n_actions=n_actions, feature_fn=feature_fn,
        )

        combined[name] = {
            'mc_eval':       mc_eval,
            'td_eval':       td_eval,
            'mc_losses':     mc['losses'],
            'td_losses':     td['losses'],
            'alphas':        mc['alphas'],
            'mc_theta_final': mc['theta_final'],
            'td_theta_final': td['theta_final'],
            'mc_thetas':     mc['thetas'],
            'td_thetas':     td['thetas'],
        }

    if prediction:
        empirical_return = mc_empirical
    else:
        empirical_return = {'mc': mc_empirical, 'td': td_empirical}

    if plot:
        _plot_summary(combined, episodes_np, empirical_return, eval_every,
                      prediction=prediction,
                      n_actions=n_actions, feature_fn=feature_fn,
                      eps_arr=eps_arr)

    return combined, empirical_return
