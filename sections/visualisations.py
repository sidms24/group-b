"""Plotting utilities.

All visualization functions for prediction, control, and decomposition analysis.
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from sections.features import compute_features_batch_jax, N_FEATURES
from sections.episodes import mse, N_ACTIONS


# Evaluation and plotting combos for prediction and control modes.

def mc_td_pred_eval(mc_thetas, td_thetas, mc_chunk_losses, td_chunk_losses,
                    eval_every=1000, n_eval=500, title_suffix="", plot=False,
                    n_actions=None, feature_fn=None):
    """Evaluate and plot prediction results (predicted vs actual return)."""
    from sections.evaluation import run_prediction_evaluation
    from sections.features import compute_features_batch_jax as _default_feat

    _n_actions = n_actions or N_ACTIONS
    _feature_fn = feature_fn or _default_feat

    eval_key = jax.random.PRNGKey(123)
    mc_actual, mc_predicted = run_prediction_evaluation(
        mc_thetas, eval_key, eval_every=eval_every,
        n_actions=_n_actions, feature_fn=_feature_fn,
    )

    eval_key = jax.random.PRNGKey(123)
    td_actual, td_predicted = run_prediction_evaluation(
        td_thetas, eval_key, eval_every=eval_every,
        n_actions=_n_actions, feature_fn=_feature_fn,
    )

    x = np.arange(len(mc_actual)) * eval_every

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(x, mc_predicted, label='MC Predicted', color='#2196F3', linestyle='--')
        axes[0].plot(x, td_actual, label='Actual', color='#228B22')
        axes[0].plot(x, td_predicted, label='TD Predicted', color='#FF5722', linestyle='--')
        axes[0].set_title(f'Predicted vs Actual Return {title_suffix}')
        axes[0].set_xlabel('Training Episode')
        axes[0].set_ylabel('Avg Return')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x, mc_actual - mc_predicted, label='MC |error|', color='#2196F3')
        axes[1].plot(x, td_actual - td_predicted, label='TD |error|', color='#FF5722')
        axes[1].set_title(f'Prediction Error {title_suffix}')
        axes[1].set_xlabel('Training Episode')
        axes[1].set_ylabel('Actual - Predicted')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(x, np.array(mc_chunk_losses), label='MC Loss', color='#2196F3')
        axes[2].plot(x, np.array(td_chunk_losses), label='TD Loss', color='#FF5722')
        axes[2].set_title(f'Training Loss {title_suffix}')
        axes[2].set_xlabel('Training Episode')
        axes[2].set_ylabel('Avg Error')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return mc_predicted, td_predicted


def mc_td_control_eval(mc_thetas, td_thetas, mc_chunk_losses, td_chunk_losses,
                       eval_every=1000, n_eval=500, baseline=75, title_suffix="", plot=False,
                       n_actions=None, feature_fn=None):
    """Evaluate and plot control results (greedy policy reward)."""
    from sections.evaluation import run_evaluation
    from sections.features import compute_features_batch_jax as _default_feat

    _n_actions = n_actions or N_ACTIONS
    _feature_fn = feature_fn or _default_feat

    eval_key = jax.random.PRNGKey(123)
    mc_eval = run_evaluation(mc_thetas, "MC", eval_key, eval_every=eval_every,
                             n_actions=_n_actions, feature_fn=_feature_fn, n_eval=n_eval)

    eval_key = jax.random.PRNGKey(123)
    td_eval = run_evaluation(td_thetas, "TD", eval_key, eval_every=eval_every,
                             n_actions=_n_actions, feature_fn=_feature_fn, n_eval=n_eval)

    x_mc = np.arange(len(mc_eval)) * eval_every
    x_td = np.arange(len(td_eval)) * eval_every

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(x_mc, mc_eval, label='MC', color='#2196F3')
        axes[0].plot(x_td, td_eval, label='TD', color='#FF5722')
        axes[0].set_xlabel('Training Episode')
        axes[0].set_ylabel(f'Avg Greedy Reward ({n_eval} eval episodes)')
        axes[0].set_title(f'MC vs TD: Greedy Policy Evaluation {title_suffix}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x_mc, np.array(mc_chunk_losses), label='MC Loss', color='#2196F3')
        axes[1].plot(x_td, np.array(td_chunk_losses), label='TD Loss', color='#FF5722')
        axes[1].set_xlabel('Training Episode')
        axes[1].set_ylabel('Avg Loss')
        axes[1].set_title(f'MC vs TD: Training Loss {title_suffix}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return mc_eval, td_eval


def _plot_summary(results, episodes_np, empirical_return, eval_every,
                  prediction=True, n_actions=None, feature_fn=None,
                  eps_arr=None):
    """Summary plot: eval curves, separate MC/TD win rate bars, alpha schedules, epsilon decay."""
    from sections.evaluation import run_winrate_vs
    from sections.policies import uniform_policy as _unif, default_player_2 as _def2
    from sections.features import compute_features_batch_jax as _default_feat

    _n_actions = n_actions or N_ACTIONS
    _feature_fn = feature_fn or _default_feat

    # 2x3 layout: [MC eval | TD eval | alpha schedules]
    #             [MC win rate bars | TD win rate bars | epsilon decay]
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
    x = np.arange(len(next(iter(results.values()))['mc_eval'])) * eval_every

    for (name, res), c in zip(results.items(), colors):
        axes[0, 0].plot(x, res['mc_eval'], label=name, color=c)
        axes[0, 1].plot(x, res['td_eval'], label=name, color=c)
        axes[0, 2].plot(episodes_np[::500], res['alphas'][::500], label=name, color=c)

    # Win rate bar charts.
    eval_key = jax.random.PRNGKey(456)
    config_names = list(results.keys())
    n_configs = len(config_names)
    x_bar = np.arange(n_configs)

    if prediction:
        w = 0.35
        mc_bars = [
            ('vs Uniform', 'mc_thetas', _unif, '#1565C0', ''),
            ('vs Default', 'mc_thetas', _def2, '#B71C1C', ''),
        ]
        td_bars = [
            ('vs Uniform', 'td_thetas', _unif, '#42A5F5', '///'),
            ('vs Default', 'td_thetas', _def2, '#EF5350', '///'),
        ]
        offsets = np.array([-0.5, 0.5]) * w
        ann_fs = 6
    else:
        w = 0.5
        mc_bars = [('vs Default', 'mc_thetas', _def2, '#2196F3', '')]
        td_bars = [('vs Default', 'td_thetas', _def2, '#FF5722', '')]
        offsets = np.array([0.0])
        ann_fs = 7

    for ax_wr, bar_spec in [(axes[1, 0], mc_bars), (axes[1, 1], td_bars)]:
        for (label, theta_key, opp_fn, color, hatch), xoff in zip(bar_spec, offsets):
            for ci, name in enumerate(config_names):
                wr = run_winrate_vs(
                    results[name][theta_key], eval_key,
                    opponent_fn=opp_fn, eval_every=eval_every,
                    n_actions=_n_actions, feature_fn=_feature_fn,
                )
                avg_wr  = float(np.mean(wr))
                best_wr = float(np.max(wr))
                best_ep = int(np.argmax(wr)) * eval_every
                bar_x   = x_bar[ci] + xoff
                ax_wr.bar(bar_x, avg_wr, width=w, color=color, hatch=hatch,
                          label=label if ci == 0 else None,
                          alpha=0.85, edgecolor='white', linewidth=0.5)
                ax_wr.text(bar_x, avg_wr + 0.01,
                           f'B:{best_wr:.2f}\n@{best_ep // 1000}k',
                           ha='center', va='bottom', fontsize=ann_fs)
        ax_wr.set_xticks(x_bar)
        ax_wr.set_xticklabels(config_names, rotation=30, ha='right', fontsize=8)
        ax_wr.set_ylabel('Avg Win Rate')
        ax_wr.set_ylim(0, 1)
        ax_wr.legend(fontsize=7)
        ax_wr.grid(True, alpha=0.3, axis='y')

    # Epsilon decay
    if eps_arr is not None:
        axes[1, 2].plot(episodes_np, eps_arr, color='#607D8B', linewidth=1.5)
        axes[1, 2].set_ylim(0, max(float(eps_arr.max()), 0.05) * 1.1)
    else:
        axes[1, 2].text(0.5, 0.5, 'eps_arr not provided',
                        ha='center', va='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Epsilon')
    axes[1, 2].set_title('Epsilon Decay Schedule')
    axes[1, 2].grid(True, alpha=0.3)

    if prediction:
        axes[0, 0].set_title('MC Predicted Value by Schedule')
        axes[0, 1].set_title('TD Predicted Value by Schedule')
        axes[1, 0].set_title('MC Win Rate: Avg bar, Best annotated')
        axes[1, 1].set_title('TD Win Rate: Avg bar, Best annotated')
    else:
        axes[0, 0].set_title('MC Greedy Reward by Schedule')
        axes[0, 1].set_title('TD Greedy Reward by Schedule')
        axes[1, 0].set_title('MC Win Rate: Avg bar, Best annotated')
        axes[1, 1].set_title('TD Win Rate: Avg bar, Best annotated')
    axes[0, 2].set_title('Alpha Schedule Shapes')

    for ax in [axes[0, 0], axes[0, 1]]:
        if isinstance(empirical_return, dict):
            ax.axhline(empirical_return['mc'], color='blue', linestyle=':',
                        alpha=0.5, label=f'MC final = {empirical_return["mc"]:.1f}')
            ax.axhline(empirical_return['td'], color='red', linestyle=':',
                        alpha=0.5, label=f'TD final = {empirical_return["td"]:.1f}')
        elif empirical_return is not None:
            ax.axhline(empirical_return, color='grey', linestyle=':',
                        label=f'Actual return = {empirical_return:.1f}')

    for ax in [axes[0, 0], axes[0, 1], axes[0, 2]]:
        ax.set_xlabel('Episode')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')
    plt.tight_layout()
    plt.show()


def plot_standalone_results(results, agent_name, eval_every, prediction,
                            empirical=None, n_actions=None, feature_fn=None, n_eval=500):
    """Plot training curves for a standalone MC or TD experiment.

    Shows a loss curve, greedy reward (control) or predicted return (prediction),
    and win rate bars for every config in results.
    """
    from sections.evaluation import run_evaluation, run_prediction_evaluation, run_winrate_vs
    from sections.policies import uniform_policy as _unif, default_player_2 as _def2
    from sections.features import compute_features_batch_jax as _default_feat

    _n_actions = n_actions or N_ACTIONS
    _feature_fn = feature_fn or _default_feat
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
    eval_key = jax.random.PRNGKey(123)
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    for (name, res), c in zip(results.items(), colors):
        x_loss = np.arange(len(res['losses'])) * eval_every
        axes[1].plot(x_loss, res['losses'], label=name, color=c)

        if prediction:
            _, predicted = run_prediction_evaluation(
                res['thetas'], eval_key, eval_every=eval_every,
                n_actions=_n_actions, feature_fn=_feature_fn,
            )
            x = np.arange(len(predicted)) * eval_every
            axes[0].plot(x, predicted, label=name, color=c)
        else:
            evals = run_evaluation(
                res['thetas'], agent_name, eval_key, eval_every=eval_every,
                n_actions=_n_actions, feature_fn=_feature_fn, n_eval=n_eval,
            )
            x = np.arange(len(evals)) * eval_every
            axes[0].plot(x, evals, label=name, color=c)

    if empirical is not None:
        label = f'Actual = {empirical:.1f}' if prediction else f'Final = {empirical:.1f}'
        axes[0].axhline(empirical, color='grey', linestyle=':', label=label)

    axes[0].set_title(f'{agent_name}: {"Predicted Return" if prediction else "Greedy Reward"}')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Avg Return' if prediction else 'Avg Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(f'{agent_name}: Training Loss')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Avg Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Win rate bar chart.
    config_names = list(results.keys())
    n_configs = len(config_names)
    x_bar = np.arange(n_configs)
    wr_key = jax.random.PRNGKey(456)

    if prediction:
        bar_specs = [
            ('vs Uniform', _unif,  '#1565C0', ''),
            ('vs Default', _def2,  '#B71C1C', '///'),
        ]
        w, offsets = 0.35, np.array([-0.5, 0.5]) * 0.35
        ann_fs = 6
    else:
        bar_specs = [('vs Default', _def2, '#2196F3' if 'MC' in agent_name else '#FF5722', '')]
        w, offsets = 0.5, np.array([0.0])
        ann_fs = 7

    for (label, opp_fn, color, hatch), xoff in zip(bar_specs, offsets):
        for ci, name in enumerate(config_names):
            wr = run_winrate_vs(
                results[name]['thetas'], wr_key,
                opponent_fn=opp_fn, eval_every=eval_every,
                n_actions=_n_actions, feature_fn=_feature_fn,
            )
            avg_wr  = float(np.mean(wr))
            best_wr = float(np.max(wr))
            best_ep = int(np.argmax(wr)) * eval_every
            bar_x   = x_bar[ci] + xoff
            axes[2].bar(bar_x, avg_wr, width=w, color=color, hatch=hatch,
                        label=label if ci == 0 else None,
                        alpha=0.85, edgecolor='white', linewidth=0.5)
            axes[2].text(bar_x, avg_wr + 0.01,
                         f'B:{best_wr:.2f}\n@{best_ep // 1000}k',
                         ha='center', va='bottom', fontsize=ann_fs)

    axes[2].set_xticks(x_bar)
    axes[2].set_xticklabels(config_names, rotation=30, ha='right', fontsize=8)
    axes[2].set_ylabel('Avg Win Rate')
    axes[2].set_ylim(0, 1)
    axes[2].set_title(f'{agent_name} Win Rate: Avg bar, Best annotated')
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


# Policy visualisation functions for inspecting learned bidding strategies.

def plot_learned_policy(theta, t_fixed=5, opp_budget=50,
                        n_actions=None, feature_fn=None):
    """Show learned bid as a function of own budget and prize value."""
    _n_actions = n_actions or N_ACTIONS
    _feature_fn = feature_fn or compute_features_batch_jax

    budgets = jnp.linspace(5, 100, 20)
    values = jnp.linspace(10, 20, 20)

    def greedy_bid(v, b):
        cands = jnp.linspace(0.0, b, _n_actions)
        phis = _feature_fn(v, t_fixed, b, opp_budget, cands)
        q_vals = phis @ theta
        return cands[jnp.argmax(q_vals)]

    bids = np.array(
        jax.vmap(lambda v: jax.vmap(lambda b: greedy_bid(v, b))(budgets))(values)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(bids, aspect='auto', origin='lower', cmap='viridis',
                    extent=[5, 100, 10, 20])
    ax.set_xlabel('Own Budget')
    ax.set_ylabel('Prize Value')
    ax.set_title(f'Learned Bid (t={t_fixed}, opp_budget={opp_budget})')
    plt.colorbar(im, label='Bid Amount')
    plt.show()


def plot_learned_policy_grid(theta, title_prefix="",
                             n_actions=None, feature_fn=None):
    """Show policy across multiple game scenarios."""
    _n_actions = n_actions or N_ACTIONS
    _feature_fn = feature_fn or compute_features_batch_jax

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    scenarios = [
        (0, 100, "Early game, equal budgets"),
        (3, 80, "Early-mid, opponent spent more"),
        (4, 50, "Mid game, equal budgets"),
        (4, 20, "Mid game, opponent poorer"),
        (8, 50, "Late game, equal budgets"),
        (9, 30, "Final round, opponent poorer"),
    ]

    budgets = jnp.linspace(5, 100, 20)
    values = jnp.linspace(10, 20, 20)

    for ax, (t_fixed, opp_budget, label) in zip(axes.flat, scenarios):
        def greedy_bid(v, b, t=t_fixed, ob=opp_budget):
            cands = jnp.linspace(0.0, b, _n_actions)
            phis = _feature_fn(v, t, b, ob, cands)
            q_vals = phis @ theta
            return cands[jnp.argmax(q_vals)]

        bids = np.array(
            jax.vmap(lambda v: jax.vmap(lambda b: greedy_bid(v, b))(budgets))(values)
        )

        im = ax.imshow(bids, aspect='auto', origin='lower', cmap='viridis',
                        extent=[5, 100, 10, 20])
        ax.set_xlabel('Own Budget')
        ax.set_ylabel('Prize Value')
        ax.set_title(f'{label}\n(t={t_fixed}, opp_b={opp_budget})')
        plt.colorbar(im, ax=ax, label='Bid')

    plt.suptitle(f'{title_prefix} Learned Bidding Strategy', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_policy_lines(theta, scenarios=None, title="",
                      n_actions=None, feature_fn=None):
    """Bid fraction curves across prize values for different budget levels."""
    _n_actions = n_actions or N_ACTIONS
    _feature_fn = feature_fn or compute_features_batch_jax

    if scenarios is None:
        scenarios = [
            (0, 100, "Early, equal"), (4, 50, "Mid, equal"),
            (8, 50, "Late, equal"), (9, 30, "Final, opp poorer"),
        ]

    values = np.linspace(10, 20, 50)
    budgets = [20, 40, 60, 80, 100]
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(budgets)))

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, (t_fixed, opp_b, label) in zip(axes, scenarios):
        for b, color in zip(budgets, cmap):
            fracs = []
            for v in values:
                cands = jnp.linspace(0, b, _n_actions)
                phis = _feature_fn(jnp.float32(v), jnp.int32(t_fixed),
                                   jnp.float32(b), jnp.float32(opp_b), cands)
                q_vals = phis @ theta
                bid = float(cands[jnp.argmax(q_vals)])
                fracs.append(bid / b)
            ax.plot(values, fracs, color=color, label=f'B={b}', linewidth=2)
        ax.set_title(label)
        ax.set_xlabel('Prize Value')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Bid Fraction (bid / budget)')
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_policy_bars(theta, title="",
                     n_actions=None, feature_fn=None):
    """Bar chart of bids for specific game situations."""
    _n_actions = n_actions or N_ACTIONS
    _feature_fn = feature_fn or compute_features_batch_jax

    cases = [
        ("Low prize\nEarly", 11, 0, 60, 50),
        ("High prize\nEarly", 19, 0, 60, 50),
        ("Low prize\nLate", 11, 8, 30, 30),
        ("High prize\nLate", 19, 8, 30, 30),
        ("Ahead\nMid", 15, 5, 70, 30),
        ("Behind\nMid", 15, 5, 30, 70),
        ("Final\nAll-in?", 18, 9, 20, 20),
    ]

    labels, bids, fracs = [], [], []
    for label, v, t, own_b, opp_b in cases:
        cands = jnp.linspace(0, own_b, _n_actions)
        phis = _feature_fn(jnp.float32(v), jnp.int32(t),
                           jnp.float32(own_b), jnp.float32(opp_b), cands)
        q_vals = phis @ theta
        bid = float(cands[jnp.argmax(q_vals)])
        labels.append(label)
        bids.append(bid)
        fracs.append(bid / own_b)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(labels))

    axes[0].bar(x, bids, color=plt.cm.viridis(np.array(fracs)))
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('Bid Amount'); axes[0].set_title(f'{title}: Absolute Bid')

    axes[1].bar(x, fracs, color=plt.cm.viridis(np.array(fracs)))
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Bid / Budget'); axes[1].set_title(f'{title}: Bid Fraction')
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.show()


def plot_feature_weights(theta, title=""):
    """Visualize which features the model weights most heavily."""
    w = np.array(theta).ravel()[:54]

    value_labels = [f'v in [{10+i},{11+i})' for i in range(10)]
    time_labels = [f't in {{{2*i},{2*i+1}}}' for i in range(5)]
    own_b_labels = [f'B in [{10*i},{10*(i+1)})' for i in range(10)]
    opp_b_labels = [f'B_opp in [{10*i},{10*(i+1)})' for i in range(10)]
    bdiff_label = ['B_diff']
    action_labels = [f'a in [{i/10:.1f},{(i+1)/10:.1f})' for i in range(10)] + ['a=1']
    interaction_labels = ['a*low_v', 'a*mid_v', 'a*hi_v',
                          'a*early', 'a*mid_t', 'a*late', 'a*Bdiff']

    all_labels = (value_labels + time_labels + own_b_labels +
                  opp_b_labels + bdiff_label + action_labels + interaction_labels)

    groups = {
        'Prize Value (10)': (0, 10),
        'Time Period (5)': (10, 15),
        'Own Budget (10)': (15, 25),
        'Opp Budget (10)': (25, 35),
        'Budget Diff (1)': (35, 36),
        'Action Bins (11)': (36, 47),
        'Interactions (7)': (47, 54),
    }

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                              gridspec_kw={'height_ratios': [1, 2]})

    colors = []
    group_colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
    for i, (name, (start, end)) in enumerate(groups.items()):
        colors.extend([group_colors[i]] * (end - start))

    axes[0].bar(range(len(w[:54])), w[:54], color=colors, edgecolor='none')
    axes[0].set_ylabel('Weight')
    axes[0].set_title(f'{title}: Feature Weights')
    axes[0].axhline(0, color='grey', linewidth=0.5)

    for name, (start, end) in groups.items():
        mid = (start + end) / 2
        axes[0].axvline(start - 0.5, color='grey', linewidth=0.5, alpha=0.3)
        axes[0].text(mid, axes[0].get_ylim()[1] * 0.95, name,
                     ha='center', fontsize=7, style='italic')
    axes[0].grid(True, alpha=0.2, axis='y')

    group_weights = {}
    for name, (start, end) in groups.items():
        group_weights[name] = w[start:end]

    max_len = max(len(v) for v in group_weights.values())
    heatmap_data = np.full((len(groups), max_len), np.nan)

    for i, (name, (start, end)) in enumerate(groups.items()):
        vals = w[start:end]
        heatmap_data[i, :len(vals)] = vals

    im = axes[1].imshow(heatmap_data, aspect='auto', cmap='RdBu_r',
                         vmin=-np.nanmax(np.abs(heatmap_data)),
                         vmax=np.nanmax(np.abs(heatmap_data)))
    axes[1].set_yticks(range(len(groups)))
    axes[1].set_yticklabels(groups.keys())
    axes[1].set_title(f'{title}: Grouped Feature Weights')

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            if not np.isnan(heatmap_data[i, j]):
                axes[1].text(j, i, f'{heatmap_data[i,j]:.2f}',
                           ha='center', va='center', fontsize=6)

    plt.colorbar(im, ax=axes[1], label='Weight (positive = increases Q)')
    plt.tight_layout()
    plt.show()


def plot_top_features(theta, top_k=20, title=""):
    """Rank the 54 features by absolute weight and plot the top-K."""
    w = np.array(theta).ravel()[:54]

    all_labels = (
        [f'v in [{10+i},{11+i})'      for i in range(10)] +
        [f't in {{{2*i},{2*i+1}}}'    for i in range(5)]  +
        [f'B in [{10*i},{10*(i+1)})'  for i in range(10)] +
        [f'Bopp in [{10*i},{10*(i+1)})' for i in range(10)] +
        ['B_diff']                                        +
        [f'a in [{i/10:.1f},{(i+1)/10:.1f})' for i in range(10)] + ['a=1'] +
        ['a*low_v', 'a*mid_v', 'a*hi_v',
         'a*early', 'a*mid_t', 'a*late', 'a*Bdiff']
    )

    group_info = [
        ('Prize Value',  (0,  10), 0),
        ('Time Period',  (10, 15), 1),
        ('Own Budget',   (15, 25), 2),
        ('Opp Budget',   (25, 35), 3),
        ('Budget Diff',  (35, 36), 4),
        ('Action Bins',  (36, 47), 5),
        ('Interactions', (47, 54), 6),
    ]
    group_colors = plt.cm.Set2(np.linspace(0, 1, 7))

    feat_color = np.empty(54, dtype=object)
    feat_group = [''] * 54
    for gname, (s, e), gi in group_info:
        for idx in range(s, e):
            feat_color[idx] = group_colors[gi]
            feat_group[idx] = gname

    top_k = min(top_k, 54)
    ranked = np.argsort(np.abs(w))[::-1][:top_k]
    ranked = ranked[::-1]

    vals    = w[ranked]
    labels  = [all_labels[i] for i in ranked]
    colors  = [feat_color[i] for i in ranked]

    fig, ax = plt.subplots(figsize=(9, max(4, top_k * 0.38)))

    bars = ax.barh(range(top_k), vals, color=colors, edgecolor='white', linewidth=0.4)
    ax.axvline(0, color='grey', linewidth=0.8)

    ax.set_yticks(range(top_k))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Weight  (positive -> raises Q-value)')
    ax.set_title(f'{title}: Top {top_k} features by |weight|', fontsize=11)

    for bar, val in zip(bars, vals):
        xpos = val + np.sign(val) * abs(ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=7)

    seen = {}
    for gi, (gname, _, g_idx) in enumerate(group_info):
        seen[gname] = plt.Rectangle((0, 0), 1, 1, color=group_colors[g_idx])
    ax.legend(seen.values(), seen.keys(), loc='lower right', fontsize=8,
              title='Feature group', title_fontsize=8)

    ax.grid(True, axis='x', alpha=0.25)
    plt.tight_layout()
    plt.show()


# MSE decomposition plot separating bias, variance, and covariance contributions.

def plot_mse_decomposition(results, config_name, phis_jax, rewards_np,
                           eval_every=1000, t=1, window=1000):
    """Plot MSE = bias^2 + Var(Q_hat) + Var(Y) - 2Cov(Y, Q_hat) decomposition."""

    Y_mc = np.cumsum(rewards_np[:, ::-1], axis=1)[:, ::-1]

    colors_parts = {'e2': '#e41a1c', 'Var(Q_hat)': '#377eb8', 'Var(Y)': '#4daf4a',
                     '-2Cov': '#984ea3', 'MSE': '#000000', 'decomp': '#FF9800'}

    res = results[config_name]
    td_thetas = jnp.array(res['td_thetas'])
    mc_thetas = jnp.array(res['mc_thetas'])

    Q_hat_td = np.zeros_like(rewards_np)
    Y_td = np.zeros_like(rewards_np)
    for c, theta_c in enumerate(td_thetas):
        s = slice(c * eval_every, (c + 1) * eval_every)
        Q_c = np.array(phis_jax[s] @ jnp.array(theta_c))
        Q_hat_td[s] = Q_c
        Y_td[s, :-1] = rewards_np[s, 1:] + Q_c[:, 1:]
        Y_td[s, -1] = rewards_np[s, -1]

    Q_hat_mc = np.zeros_like(rewards_np)
    for c, theta_c in enumerate(mc_thetas):
        s = slice(c * eval_every, (c + 1) * eval_every)
        Q_hat_mc[s] = np.array(phis_jax[s] @ jnp.array(theta_c))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # MC MSE decomposition into bias, variance, and covariance terms.
    bias_mc = (pd.Series(Y_mc[:, t]).rolling(window).mean()
               - pd.Series(Q_hat_mc[:, t]).rolling(window).mean()) ** 2
    varQ_mc = pd.Series(Q_hat_mc[:, t]).rolling(window).var()
    varY_mc = pd.Series(Y_mc[:, t]).rolling(window).var()
    prod_mc = pd.Series(Y_mc[:, t] * Q_hat_mc[:, t]).rolling(window).mean()
    meanY_mc = pd.Series(Y_mc[:, t]).rolling(window).mean()
    meanQ_mc = pd.Series(Q_hat_mc[:, t]).rolling(window).mean()
    cov_mc = prod_mc - meanY_mc * meanQ_mc
    decomp_mc = bias_mc + varQ_mc + varY_mc - 2 * cov_mc
    mse_mc = pd.Series((Q_hat_mc[:, t] - Y_mc[:, t]) ** 2).rolling(window).mean()

    axes[0, 0].plot(bias_mc, label='e^2', color=colors_parts['e2'], alpha=0.8)
    axes[0, 0].plot(varQ_mc, label='Var(Q_hat)', color=colors_parts['Var(Q_hat)'], alpha=0.8)
    axes[0, 0].plot(varY_mc, label='Var(Y_MC)', color=colors_parts['Var(Y)'], alpha=0.8)
    axes[0, 0].plot(mse_mc, label='MSE', color=colors_parts['MSE'], linewidth=2, linestyle='--')
    axes[0, 0].plot(-2 * cov_mc, label='-2Cov(Y_MC, Q_hat)', color=colors_parts['-2Cov'], alpha=0.8)
    axes[0, 0].plot(decomp_mc, label='Decomposed MSE', color=colors_parts['decomp'], linewidth=1, linestyle=':')
    axes[0, 0].set_title(f'MC MSE Decomposition (Round {t}): {config_name}')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # TD MSE decomposition using the bootstrapped TD targets.
    bias_td = (pd.Series(Y_td[:, t]).rolling(window).mean()
               - pd.Series(Q_hat_td[:, t]).rolling(window).mean()) ** 2
    varQ_td = pd.Series(Q_hat_td[:, t]).rolling(window).var()
    varY_td = pd.Series(Y_td[:, t]).rolling(window).var()
    prod = pd.Series(Y_td[:, t] * Q_hat_td[:, t]).rolling(window).mean()
    meanY = pd.Series(Y_td[:, t]).rolling(window).mean()
    meanQ = pd.Series(Q_hat_td[:, t]).rolling(window).mean()
    cov_td = prod - meanY * meanQ
    decomp_td = bias_td + varQ_td + varY_td - 2 * cov_td
    mse_td = pd.Series((Q_hat_td[:, t] - Y_td[:, t]) ** 2).rolling(window).mean()

    axes[0, 1].plot(bias_td, label='e^2', color=colors_parts['e2'], alpha=0.8)
    axes[0, 1].plot(varQ_td, label='Var(Q_hat)', color=colors_parts['Var(Q_hat)'], alpha=0.8)
    axes[0, 1].plot(varY_td, label='Var(Y_TD)', color=colors_parts['Var(Y)'], alpha=0.8)
    axes[0, 1].plot(-2 * cov_td, label='-2Cov(Y_TD, Q_hat)', color=colors_parts['-2Cov'], alpha=0.8)
    axes[0, 1].plot(mse_td, label='MSE', color=colors_parts['MSE'], linewidth=2, linestyle='--')
    axes[0, 1].plot(decomp_td, label='Decomposed MSE', color=colors_parts['decomp'], linewidth=1, linestyle=':')
    axes[0, 1].set_title(f'TD MSE Decomposition (Round {t}): {config_name}')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Overlay smoothed predicted and actual returns for visual comparison.
    mc_pred = pd.Series(Q_hat_mc[:, t]).rolling(window).mean()
    td_pred = pd.Series(Q_hat_td[:, t]).rolling(window).mean()
    actual = pd.Series(Y_mc[:, t]).rolling(window).mean()

    axes[1, 1].plot(actual, label='Actual return', color='grey', linewidth=2)
    axes[1, 1].plot(mc_pred, label='MC Q_hat', color='#2196F3', alpha=0.8)
    axes[1, 1].plot(td_pred, label='TD Q_hat', color='#FF5722', alpha=0.8)
    axes[1, 1].set_title(f'Predicted vs Actual Return (Round {t})')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # Numerical error between decomposed MSE and directly computed MSE.
    diff_mc = np.abs(decomp_mc - mse_mc)
    diff_td = np.abs(decomp_td - mse_td)

    axes[1, 0].plot(diff_mc, label='MC: decomposed - MSE', color='#2196F3', alpha=0.8)
    axes[1, 0].plot(diff_td, label='TD: decomposed - MSE', color='#FF5722', alpha=0.8)
    axes[1, 0].axhline(0, color='grey', linestyle=':', alpha=0.5)
    axes[1, 0].set_title(f'Decomposed MSE - Actual MSE (Round {t})')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')

    plt.tight_layout()
    plt.show()
