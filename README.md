# README — Group B

**Advanced Machine Learning, EFDS 2025–26**

Dhruv Syam, Jiaqi Chen, Said Musse — March 6, 2026

---

## Folder Structure

```
├── Notebooks/
│   ├── prediction.ipynb                # Q2.3.2 MC and TD Prediction
│   ├── control.ipynb                   # Q2.3.3 MC and TD Control
│   ├── tournament_agent_final.ipynb    # Q2.4 Tournament Agent (NFSP)
│   └── ppo_sac.ipynb                   # Experimenting With PPO
├── sections/
│   ├── __init__.py
│   ├── environment.py
│   ├── policies.py
│   ├── features.py
│   ├── episodes.py
│   ├── mc.py
│   ├── td.py
│   ├── sarsa.py
│   ├── schedules.py
│   ├── trainer.py
│   ├── evaluation.py
│   └── visualisations.py
├── rlagent/
│   ├── agentB.py                       # Tournament agent
│   └── weights.npz
├── said_weight/
│   └── ppo_sac_weights.npz            # Said's agent weights (for NFSP training)
└── requirements.txt
```

## Dependencies

All packages are listed in `requirements.txt` and installed automatically by the first cell of each notebook.

| Package      | Version  | Purpose                                                        |
|--------------|----------|----------------------------------------------------------------|
| `numpy`      | ≥1.24.0  | Array operations, NumPy-only tournament agent inference        |
| `jax`        | ≥0.4.20  | Automatic differentiation, JIT compilation, `vmap` vectorisation |
| `jaxlib`     | ≥0.4.20  | Backend for JAX (CPU/GPU/TPU)                                  |
| `matplotlib` | ≥3.7.0   | All plots: training curves, heatmaps, bar charts               |
| `pandas`     | ≥2.0.0   | Rolling-window statistics in MSE decomposition analysis        |
| `tqdm`       | ≥4.65.0  | Progress bars during training and hyperparameter sweeps        |
| `pyyaml`     | ≥6.0     | YAML configuration parsing (JAX internal dependency)           |
| `flax`       | ≥0.8.0   | Neural network layers for the NFSP tournament agent (Q2.4)     |
| `optax`      | ≥0.1.7   | Optimisers (Adam, cosine schedule) for NFSP training (Q2.4)    |

## Running Notebooks

All notebooks are designed for **Google Colab**. Each notebook's first cell clones the repo and installs dependencies:

```python
!git clone https://github.com/sidms24/group-b.git
%cd group-b
!pip install -q -r requirements.txt
```

### Q2.3.2: Prediction (`prediction.ipynb`)

Trains MC and TD(0) agents in prediction mode with MSE and Huber losses. Sweeps learning rate schedules (inverse decay, cosine annealing with warmup). Outputs: predicted vs actual return curves, MSE decomposition (bias, variance, covariance), and training loss plots.

Runtime: ~3–5 minutes on Colab V6e TPU.

### Q2.3.3: Control (`control.ipynb`)

Runs a hyperparameter sweep over {MC, TD(0)} × {54-dim, 65-dim, tile-coded, combined} feature representations. For each combination, sequentially optimises batch size, learning rate, and epsilon schedules. Outputs: reward curves, win rate evaluations vs Default Player 2, strategy heatmaps, and summary tables.

Runtime: ~20–30 minutes on Colab V6e TPU.

### Q2.4: Tournament Agent (`tournament_agent_final.ipynb`)

Trains the tournament agent via Neural Fictitious Self-Play (NFSP) with PPO best-response and a mixture-of-3-Betas continuous policy. Uses diverse asymmetric opponents (55% self-play + league, 15% Said's agent, 30% synthetic opponents). Exports the average strategy (`pi_avg`) to `rlagent/` for tournament deployment.

Requires `said_weight/ppo_sac_weights.npz` (included in the repo).

Runtime: ~15–20 minutes on Colab GPU.

### Tournament Agent Usage

```python
from rlagent.agentB import policyB

bid = policyB(t, v, B_own, B_opp)
```
