# Energy Trading Quant Tools

A modular Python toolkit for quantitative development in energy trading: price models, risk, backtesting utilities, and data-cleaning helpers.

## What is included (initial scaffold)

- `src/energytrading/`: Python package
  - `models/`: quantitative models (starting with Ornstein-Uhlenbeck mean reversion)
  - `data/`: dataset utilities and feature helpers (placeholder)
  - `risk/`: risk metrics (placeholder)
- `docs/`: lightweight markdown docs
- `tests/`: pytest unit tests

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Example: simulate a mean-reverting OU spot price

```python
from energytrading.models.ou import OUParams, simulate_ou

params = OUParams(mu=50.0, theta=1.2, sigma=8.0, dt=1/252)
path = simulate_ou(x0=50.0, n_steps=252, params=params, seed=7)
print(path[:5])
```

## Design principles

- Modular: models are self-contained modules with explicit parameter objects.
- Adjustable: everything important is parameterized and documented.
- Usable: includes tests and examples, avoids hidden global state.

## Roadmap ideas

- Seasonality + mean reversion (OU + deterministic seasonal term)
- Spike models for power (jump / regime-switching)
- Multi-factor forward curve models (e.g., Schwartz-Smith style)
- Risk: VaR/CVaR, scenario utilities
- Backtesting scaffolds for signals and hedging

Note: Models are educational and need validation for production use.