# Random Seed Influence Study

Empirical study of how random seed selection affects model performance, variance, and reproducibility across ML experiments.

## Problem

Practitioners often pick a single seed without measuring sensitivity. This understates variance and overstates model stability.

## Approach

Run repeated experiments across multiple seeds and compare:

- Train/validation metric distributions
- Convergence behavior
- Best vs median performance gap

See `RandomSeed_influence.ipynb` for the full analysis. Related work also appears in [Machine-and-Influence-learning](https://github.com/baban9/Machine-and-Influence-learning) (`11_RandomSeed_influence.ipynb`).

## Reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Set seeds explicitly at the top of each experiment cell:

```python
import random, numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
```

## Tech stack

Python, Jupyter, NumPy, scikit-learn

## Limitations and next steps

- Log all seeds and hyperparameters to MLflow or W&B
- Extend study to deep learning frameworks (PyTorch, TensorFlow)
- Report confidence intervals, not single-run metrics
