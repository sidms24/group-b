"""Political campaign bidding environment.

Class-based environment for simulation and evaluation.
"""

import numpy as np


class PoliticalCampaignEnv:

    def __init__(self, T=10, B0=100.0, v_min=10, v_max=20):
        self.T = T
        self.B0 = B0
        self.v_min = v_min
        self.v_max = v_max

    def simulate(self, t=0, B1=None, B2=None, policy1=None, policy2=None, n_periods=None):

        if B1 is None: B1 = self.B0
        if B2 is None: B2 = self.B0
        if n_periods is None: n_periods = self.T - t

        if t + n_periods > self.T:
            raise ValueError(
                f"Cannot start at t={t} and simulate {n_periods} periods "
                f"(would exceed T={self.T})"
            )

        rewards = np.zeros((2, n_periods))
        budgets = np.zeros((2, n_periods))

        b1, b2 = B1, B2

        for i in range(n_periods):
            current_t = t + i
            budgets[0, i] = b1
            budgets[1, i] = b2

            v = np.random.uniform(self.v_min, self.v_max)

            bid1 = np.clip(policy1(current_t, v, b1, b2), 0, b1)
            bid2 = np.clip(policy2(current_t, v, b2, b1), 0, b2)

            if bid1 > bid2:
                rewards[0, i] = v
            elif bid2 > bid1:
                rewards[1, i] = v
            else:
                rewards[0, i] = v / 2
                rewards[1, i] = v / 2

            b1 -= bid1
            b2 -= bid2

        return t + n_periods, rewards, budgets

    def play_full_game(self, policy1, policy2, n_games=1000):
        total_rewards = np.zeros((2, n_games))
        for g in range(n_games):
            _, rewards, _ = self.simulate(policy1=policy1, policy2=policy2)
            total_rewards[:, g] = rewards.sum(axis=1)
        return total_rewards


def simulate_game(t=0, B1=100.0, B2=100.0, policy1=None, policy2=None, n_periods=10):

    env = PoliticalCampaignEnv()
    return env.simulate(t=t, B1=B1, B2=B2,
                        policy1=policy1, policy2=policy2,
                        n_periods=n_periods)
