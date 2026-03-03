# ⚡ Energy Trading Quant Tools

[![CI](https://github.com/gencersarp/energy-trading-quant-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/gencersarp/energy-trading-quant-tools/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An institutional-grade Python framework for **quantitative research and algorithmic trading in energy and commodity markets**. 

Unlike standard equity markets, energy markets (Power, Natural Gas) exhibit extreme non-linearities: massive positive kurtosis (heavy tails), aggressive mean-reversion, physical seasonality, and discontinuous price spikes. This library provides the mathematical and infrastructural primitives to model, trade, and manage risk around these physical realities.

---

## 🏗️ Architecture & Core Capabilities

The framework is divided into modular, highly-vectorized components designed for both fast prototyping and robust out-of-sample research.

### 1. Stochastic Pricing Models (`src/energytrading/models/`)
*   **Ornstein-Uhlenbeck (OU):** Mean-reverting continuous-time processes with MLE/OLS calibration for parameter extraction ($\mu$, $\theta$, $\sigma$).
*   **Schwartz-Smith 2-Factor:** Captures short-term spot deviations and long-term equilibrium shifts with Kalman Filter compatibility.
*   **Regime-Switching (HMM):** 2-state Gaussian Hidden Markov Models (Base vs. Spike) to dynamically probability-weight market regimes.
*   **Merton Jump-Diffusion:** Continuous-time diffusion overlaid with Poisson-driven discontinuous jumps for options pricing and scenario simulation.
*   **Forward Curve Bootstrapping:** Arbitrage-free daily curve construction from coarse Monthly/Quarterly block contracts using Cubic Spline interpolation.

### 2. Portfolio & Risk Management (`src/energytrading/portfolio/` & `risk/`)
*   **Extreme Value Theory (EVT):** Peaks-Over-Threshold (POT) modeling via Generalized Pareto Distributions (GPD) for accurate Tail Risk and Conditional VaR (Expected Shortfall).
*   **Regime-Conditional Capital Allocation:** Dynamic fractional Kelly sizing and Volatility Targeting conditioned on HMM state probabilities to prevent ruin during volatility expansions.
*   **Stress Testing:** Scenario engines to simulate catastrophic systemic grid failures (e.g., Winter Storm Uri).

### 3. Physical Asset Optimization (`src/energytrading/pricing/`)
*   **Battery Storage LP:** A `PuLP`-powered Linear Programming solver to optimize deterministic battery dispatch schedules. Maximizes arbitrage revenue subject to capacity, degradation costs, and round-trip efficiency constraints. 

### 4. Backtesting & Execution Realism (`src/energytrading/backtest/` & `validation/`)
*   **Vectorized Engine:** Fast Pandas-based backtester with vectorized log-returns and dynamically updating transaction costs.
*   **Execution Slippage:** Implements mathematical market impact modeling (Bid/Ask basis point spreads and Square-Root liquidity impact laws).
*   **Walk-Forward Validation:** Strict out-of-sample rolling train/test cross-validation engine to completely eliminate lookahead bias and data leakage.

### 5. Data Ingestion (`src/energytrading/data/`)
*   **EIA & ENTSO-E Clients:** Native `requests`-based wrappers to pull retail sales, electricity pricing, and European Day-Ahead capacity.
*   **Granularity Alignment:** Tools to safely forward-fill and merge sparse daily gas contracts against dense hourly power prices.

---

## 🚀 Installation & Quickstart

Clone the repository and install the package with developer dependencies:

```bash
git clone https://github.com/gencersarp/energy-trading-quant-tools.git
cd energy-trading-quant-tools

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package and dependencies (PuLP, hmmlearn, yfinance, etc.)
pip install -e ".[dev]"
```

### Example 1: Fitting an OU Process to Extract Mean Reversion

```python
import numpy as np
from energytrading.models.ou import fit_ou_mle, simulate_ou, OUParams

# 1. Simulate a true OU path
true_params = OUParams(mu=50.0, theta=1.5, sigma=5.0, dt=1.0)
mock_prices = simulate_ou(x0=50.0, n_steps=8000, params=true_params)

# 2. Extract parameters blindly from the time series using MLE
fit = fit_ou_mle(mock_prices, dt=1.0)
print(f"Extracted Mu: {fit.params.mu:.2f}")
print(f"Extracted Theta (Speed): {fit.params.theta:.2f}")
print(f"Extracted Sigma (Vol): {fit.params.sigma:.2f}")
```

### Example 2: Optimizing Battery Dispatch (Linear Programming)

```python
import numpy as np
from energytrading.pricing.battery import BatteryStorageOptimizer

# Spot prices over 24 hours
prices = np.array([20, 15, 10, 10, 15, 30, 80, 150, 120, 60, 40, 30, 
                   35, 45, 60, 90, 200, 250, 180, 100, 50, 30, 25, 20])

optimizer = BatteryStorageOptimizer(
    capacity=100.0,       # 100 MWh 
    max_charge=25.0,      # 25 MW charge rate
    max_discharge=25.0,   # 25 MW discharge rate
    efficiency=0.85,      # 85% round-trip
    degradation_cost=2.5  # $2.50 per MWh cycled
)

schedule = optimizer.optimize_schedule(prices)
print("Optimal Dispatch Schedule (MW):", schedule)
```

---

## 📖 PM Defense & Quant Research Notes

For insights into the theoretical justifications of this library (e.g., why Sharpe Ratio fails in power markets, the limitations of deterministic LP battery solvers, and how to defend this architecture in a PM interview), please read the included whitepaper:

👉 **[docs/PM_DEFENSE.md](docs/PM_DEFENSE.md)**

---

## 🧪 Testing and CI/CD

This repository enforces strict structural guidelines. Code is validated against `ruff` for linting and `pytest` for mathematical correctness. 

To run the test suite locally:
```bash
ruff check .
pytest
```

*Disclaimer: These models are provided for educational and benchmarking purposes. Production implementation requires live exchange connectivity, continuous data cleaning, and real-time execution engines.*