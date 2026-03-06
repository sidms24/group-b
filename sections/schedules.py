"""Learning rate and exploration schedules.

All schedule functions take the episode number and return a scalar value.
"""

import numpy as np


# Learning rate schedules that map episode number to a step size.

def schedule_decay(ep, a0=1e-4, d=1e-5):
    """Inverse decay: a0 / (1 + d * ep)."""
    return a0 / (1.0 + d * ep)


def schedule_cosine(ep, n, amax=5e-4, amin=1e-5):
    """Cosine annealing from amax to amin over n episodes."""
    return amin + 0.5 * (amax - amin) * (1 + np.cos(np.pi * ep / n))


def schedule_cosine_warmup(ep, n, amax=5e-4, amin=1e-5, wf=0.1):
    """Cosine annealing with linear warmup for the first wf fraction of training."""
    ws = int(n * wf)
    if ep < ws:
        return amin + (amax - amin) * ep / ws
    return amin + 0.5 * (amax - amin) * (1 + np.cos(np.pi * (ep - ws) / (n - ws)))


def schedule_eps_linked(ep, n, eps, amax=5e-4, amin=1e-5):
    """Learning rate linked to epsilon: higher alpha when epsilon is low."""
    return max(amin, amax * (1.0 - eps))


def schedule_eps_linked_faster(ep, n, eps, amax=5e-4, amin=1e-5, coeff=1.5):
    """Faster epsilon-linked schedule with power coefficient."""
    return max(amin, amax * (1.0 - eps) ** coeff)


# Exploration schedules that map episode number to an epsilon value.

def epsilon_schedule_exp(ep, es=1.0, ee=0.1, ed=0.99997):
    """Exponential epsilon decay from es to ee."""
    return max(ee, es * (ed ** ep))


def epsilon_schedule_cosine(ep, n, emax=1.0, emin=0.1):
    """Cosine epsilon decay from emax to emin over n episodes."""
    return emin + 0.5 * (emax - emin) * (1 + np.cos(np.pi * ep / n))
