# Energy Trading Quant Platform

[![CI](https://github.com/gencersarp/energy-trading-quant-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/gencersarp/energy-trading-quant-tools/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An institutional-grade, modular quantitative platform for energy market research, trading, and risk management. Built to rival infrastructure at commodity trading houses, power market desks, quant hedge funds, and algorithmic trading firms.

Unlike standard equity markets, energy markets (Power, Natural Gas) exhibit extreme non-linearities: massive positive kurtosis (heavy tails), aggressive mean-reversion, physical seasonality, and discontinuous price spikes. This library provides the mathematical and infrastructural primitives to model, trade, and manage risk around these physical realities.

---

## Architecture Overview

```
src/energytrading/
├── models/                   # Stochastic & ML pricing models
├── risk/                     # Risk metrics, stress tests, Monte Carlo
├── portfolio/                # Portfolio optimization & hedging
├── signals/                  # Alpha signal generation
├── backtest/                 # Backtesting engines & execution algorithms
├── pricing/                  # Energy derivative pricing
├── features/                 # Feature engineering & store
├── data/                     # Data ingestion & normalization
├── execution/                # Live execution engine
├── simulation/               # Monte Carlo & scenario simulation
├── optimization/             # Physical asset optimization (gas, hydro, MILP)
├── powergrid/                # DC power flow, LMP, GNN
├── ml/                       # Model registry, experiment tracker, hyperopt
├── rl/                       # Reinforcement learning environments & agents
├── streaming/                # Kafka, Redis, WebSocket
├── weather/                  # Weather ingestion, forecasting, derivatives
├── regimes/                  # Market regime detection & turbulence
├── market_microstructure/    # Order book, market impact, microstructure
├── validation/               # Walk-forward validation
├── storage/                  # Bitemporal data storage
├── strategy/                 # Virtual bidding strategies
├── api/                      # FastAPI REST endpoints
└── dashboard/                # Streamlit trader dashboard
```

---

## Core Modules

### 1. Stochastic Pricing Models (`models/`)

| Model | Description |
|-------|-------------|
| `ou.py` | Ornstein-Uhlenbeck mean-reversion (exact MLE calibration) |
| `schwartz_smith.py` | Two-factor Schwartz-Smith commodity model |
| `jump_diffusion.py` | Merton jump-diffusion for price spikes |
| `regime_switching.py` | Hidden Markov Model regime switching |
| `seasonal.py` | Fourier seasonal decomposition |
| `forward_curve.py` | Block bootstrapping + cubic spline curve building |
| `lgbm_forecaster.py` | LightGBM intraday price forecaster |
| `arima.py` | ARIMA/SARIMA/GARCH/EGARCH with auto-order selection (AIC) |
| `kalman.py` | Kalman filter + RTS smoother, Extended KF, dynamic beta tracker |
| `cointegration.py` | Engle-Granger, Johansen, VAR, ECM, spread trading model |
| `heston.py` | Heston stochastic volatility with Gil-Pelaez CF inversion |
| `deep_learning.py` | LSTM, Temporal CNN, Transformer, TFT (pure numpy) |

### 2. Risk Management (`risk/`)

| Module | Description |
|--------|-------------|
| `metrics.py` | Historical/parametric VaR, CVaR, Expected Shortfall |
| `evt.py` | Extreme Value Theory — Peaks-Over-Threshold / GPD tail risk |
| `greeks.py` | Delta, gamma, vega, theta for energy options |
| `scenarios.py` | Historical scenario analysis |
| `copula.py` | Gaussian, Student-T, Clayton, Gumbel copulas; tail dependence |
| `stress_testing.py` | Polar vortex, gas supply shock, carbon shock, reverse stress test |
| `monte_carlo.py` | Correlated path simulation, LVAR, component ES decomposition |

### 3. Portfolio Optimization (`portfolio/`)

| Module | Description |
|--------|-------------|
| `regime_allocation.py` | Regime-weighted Kelly + volatility targeting |
| `optimization.py` | Mean-Variance (SLSQP), Risk Parity (ERC), CVaR-LP, Kelly criterion |
| `hedging.py` | Min-variance hedge, spark spread hedging (Margrabe), weather hedging |

### 4. Signal Generation (`signals/`)

| Module | Description |
|--------|-------------|
| `statistical.py` | Volatility breakout, weather deviation |
| `breakout.py` | Price breakout signals |
| `microstructure.py` | Order flow, bid-ask bounce |
| `ml_forecasting.py` | ML-based price direction forecasting |
| `regime.py` | Regime-conditioned signals |
| `renewables.py` | Wind/solar generation-based signals |
| `wind_shock.py` | Wind shock event signals |
| `ensemble.py` | SignalStack, MetaLabeler (Lopez de Prado), BayesianSignalAggregator |
| `calendar_spread.py` | Calendar/seasonal spreads, cross-commodity arb, vol arbitrage |

### 5. Backtesting (`backtest/`)

| Module | Description |
|--------|-------------|
| `engine.py` | Vectorized backtesting with lookahead prevention |
| `event_driven.py` | Event-driven backtest engine |
| `execution_realism.py` | Realistic execution with slippage models |
| `gate_closure_engine.py` | Gate closure–aware energy market backtester |
| `strategies.py` | Pre-built strategy templates |
| `twap_vwap.py` | TWAP, VWAP, POV, IS optimizer, Smart Order Router |

### 6. Feature Engineering (`features/`)

| Module | Description |
|--------|-------------|
| `intraday_features.py` | Order book features, fundamental alignment (polars) |
| `feature_store.py` | Feature registry, versioning, caching, materialization |
| `energy_features.py` | Spark/dark/clean spreads, heat rates, reserve margins, duck curve |
| `transforms.py` | Fourier, Haar wavelet, rolling statistics, spike detection |

### 7. Data Layer (`data/`)

| Module | Description |
|--------|-------------|
| `clients.py` | Generic market data clients |
| `ingestors.py` | yfinance ingestor, time series alignment |
| `streams.py` | Real-time data streaming |
| `synthetic.py` | Synthetic data generation |
| `normalizer.py` | Timezone alignment, normalization, contract rolling, curve stitching |
| `eia_client.py` | EIA API v2: gas prices/storage, electricity, crude (synthetic fallback) |
| `entsoe_client.py` | ENTSO-E: day-ahead prices, load, generation, cross-border flows |

### 8. Simulation Engine (`simulation/`)

| Module | Description |
|--------|-------------|
| `monte_carlo.py` | GBM, OU, Merton, Heston, correlated paths; antithetic, QMC Sobol |
| `scenarios.py` | Polar vortex, heatwave, wind drought, negative prices, black swan |

### 9. Physical Asset Optimization (`optimization/`)

| Module | Description |
|--------|-------------|
| `gas_storage.py` | Backward-induction DP + LP for gas storage; extrinsic value via MC |
| `hydro.py` | Hydro dispatch LP + stochastic SAA + cascade systems |
| `milp.py` | Unit commitment MILP with binary on/off, ramp rates, startup costs |

### 10. Power Grid (`powergrid/`)

| Module | Description |
|--------|-------------|
| `power_flow.py` | DC power flow solver, N-1 contingency analysis, grid topology |
| `lmp.py` | LMP decomposition (energy + congestion + loss), FTR valuation |
| `gnn.py` | Graph Neural Network for nodal price forecasting |

### 11. ML Infrastructure (`ml/`)

| Module | Description |
|--------|-------------|
| `model_registry.py` | Model versioning, staging (dev/staging/production), promotion |
| `experiment_tracker.py` | MLflow-compatible JSON experiment tracking |
| `ensemble.py` | IC-weighted ensemble, meta-labeling, Bayesian model averaging |
| `hyperopt.py` | Grid, random, TPE, CMA-ES hyperparameter optimization |

### 12. Reinforcement Learning (`rl/`)

| Module | Description |
|--------|-------------|
| `environment.py` | EnergyMarketEnv, BatteryStorageEnv, SpreadTradingEnv (gym-compatible) |
| `agents.py` | Random, RuleBased, LinearQ, PolicyGradient/REINFORCE, PPO (pure numpy) |

### 13. Real-Time Streaming (`streaming/`)

| Module | Description |
|--------|-------------|
| `kafka_handler.py` | Kafka producer/consumer, stream processor, in-memory fallback |
| `redis_handler.py` | Redis pub/sub, cache layer with TTL, in-memory fallback |
| `websocket_handler.py` | WebSocket price feed, real-time signal engine |

### 14. Weather (`weather/`)

| Module | Description |
|--------|-------------|
| `ingestion.py` | OpenMeteo API client, multi-station pipeline, HDD/CDD computation |
| `forecasting.py` | Weather-load GBM model, wind power curve, solar irradiance |
| `derivatives.py` | HDD/CDD option pricing (MC), swaps, burn analysis, implied vol |

### 15. Regime Detection (`regimes/`)

| Module | Description |
|--------|-------------|
| `detector.py` | HMM, threshold, binary segmentation change-point, Markov chain |
| `turbulence.py` | Turbulence index, Absorption Ratio (PCA), stress composite (0-1) |

### 16. Market Microstructure (`market_microstructure/`)

| Module | Description |
|--------|-------------|
| `order_book.py` | Full LOB (price-time priority), Hawkes-like Poisson order arrivals |
| `impact.py` | Linear/sqrt impact, Almgren-Chriss optimal execution, Roll spread, PIN |

---

## Installation & Quickstart

```bash
git clone https://github.com/gencersarp/energy-trading-quant-tools.git
cd energy-trading-quant-tools

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Optional extras
pip install -e ".[redis]"     # Redis streaming
pip install -e ".[full]"      # All optional deps
```

### Environment Variables

```bash
EIA_API_KEY=<your_eia_key>          # EIA Open Data API (optional, falls back to synthetic)
ENTSOE_TOKEN=<your_entsoe_token>    # ENTSO-E Transparency (optional)
REDIS_URL=redis://localhost:6379    # Redis (optional)
KAFKA_BROKERS=localhost:9092        # Kafka (optional)
```

---

## Code Examples

### OU Mean-Reversion Calibration

```python
from energytrading.models.ou import OUParams, simulate_ou, fit_ou_mle
import numpy as np

params = OUParams(kappa=2.0, mu=50.0, sigma=5.0)
prices = simulate_ou(params, S0=50.0, T=1.0, n_steps=252)
fitted = fit_ou_mle(prices)
print(f"kappa={fitted.kappa:.3f}, mu={fitted.mu:.3f}, sigma={fitted.sigma:.3f}")
```

### Gas Storage Optimization (Dynamic Programming)

```python
from energytrading.optimization.gas_storage import GasStorageParams, GasStorageDP
import numpy as np

params = GasStorageParams(
    max_capacity=100.0, min_capacity=10.0,
    max_injection=10.0, max_withdrawal=15.0,
    injection_cost=0.5, withdrawal_cost=0.3,
)
optimizer = GasStorageDP(params, n_inventory_steps=20)
prices = np.random.normal(3.0, 0.5, 365)
value = optimizer.solve(prices, initial_inventory=50.0)
print(f"Storage option value: ${value:.2f}")
```

### Battery Storage (LP)

```python
import numpy as np
from energytrading.pricing.battery import BatteryStorageOptimizer

prices = np.array([20, 15, 10, 10, 15, 30, 80, 150, 120, 60, 40, 30,
                   35, 45, 60, 90, 200, 250, 180, 100, 50, 30, 25, 20])

optimizer = BatteryStorageOptimizer(
    capacity=100.0, max_charge=25.0, max_discharge=25.0,
    efficiency=0.85, degradation_cost=2.5,
)
schedule = optimizer.optimize_schedule(prices)
print("Optimal Dispatch Schedule (MW):", schedule)
```

### Calendar Spread Signal

```python
from energytrading.signals.calendar_spread import CalendarSpreadSignal
import pandas as pd, numpy as np

idx = pd.date_range("2023-01-01", periods=500, freq="B")
front = pd.Series(50 + np.random.randn(500).cumsum() * 0.5, index=idx)
back  = pd.Series(48 + np.random.randn(500).cumsum() * 0.5, index=idx)

signal = CalendarSpreadSignal(entry_z=1.5).compute_signal(front, back)
print(signal.value_counts())
```

### Monte Carlo Portfolio Risk

```python
from energytrading.risk.monte_carlo import CorrelatedPathSimulator, MCRiskEngine
import numpy as np

sim = CorrelatedPathSimulator(
    mu=np.array([0.05, 0.03]),
    sigma=np.array([0.2, 0.15]),
    corr=np.array([[1.0, 0.6], [0.6, 1.0]]),
)
engine = MCRiskEngine(sim, n_simulations=10_000)
paths = engine.simulate_portfolio(
    weights=np.array([0.6, 0.4]), S0=np.array([100.0, 80.0]),
    T=1.0, n_steps=252,
)
print(f"95% VaR:  {engine.var(paths):.4f}")
print(f"95% CVaR: {engine.cvar(paths):.4f}")
```

### EIA & ENTSO-E Data

```python
from energytrading.data.eia_client import EIAClient
from energytrading.data.entsoe_client import ENTSOEClient

# Falls back to synthetic data without API key
eia = EIAClient()
gas = eia.get_gas_spot_price("2023-01-01", "2024-01-01")

entsoe = ENTSOEClient()
da_prices = entsoe.get_day_ahead_prices("DE", "2024-01-01", "2024-01-31")
```

### TWAP / VWAP Execution

```python
from energytrading.backtest.twap_vwap import TWAPExecutor, VWAPExecutor, IS_Optimizer
import pandas as pd, numpy as np

idx = pd.date_range("2024-01-01 09:00", periods=100, freq="5min")
prices = pd.Series(50 + np.random.randn(100).cumsum() * 0.1, index=idx)

twap_result = TWAPExecutor(n_slices=10).execute(10_000, prices)
print(f"TWAP: {twap_result.slippage_bps:.2f} bps")

is_result = IS_Optimizer(risk_aversion=0.01).execute(10_000, prices)
print(f"IS:   {is_result.slippage_bps:.2f} bps")
```

### PPO Reinforcement Learning Agent

```python
from energytrading.rl.environment import EnergyMarketEnv
from energytrading.rl.agents import PPOAgent, AgentEvaluator
import numpy as np

prices = np.random.normal(50, 10, 500)
env = EnergyMarketEnv(prices)
agent = PPOAgent(state_dim=env.obs_dim, action_dim=1)

for episode in range(10):
    state = env.reset()
    done = False
    while not done:
        action, log_prob = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.store(state, action, reward, log_prob)
        state = next_state
    agent.update()
```

### Starting the API Server

```bash
uvicorn energytrading.api.main:app --reload --port 8000
# Available at http://localhost:8000/docs
```

---

## Running Tests

```bash
pytest tests/ -v --cov=energytrading --cov-report=term-missing
```

---

## Design Principles

1. **No deleted functionality** — all v0.4 modules preserved intact; this is a purely additive expansion
2. **Graceful degradation** — every module works without optional dependencies (Redis, Kafka, torch, statsmodels, requests) via in-memory or numpy fallbacks
3. **Numpy-first ML** — deep learning models implemented in pure numpy to avoid mandatory torch/TensorFlow dependency
4. **Energy-domain specificity** — models, signals, and data clients tuned for gas, power, and commodity markets
5. **Production-ready patterns** — model registry with versioning, MLflow-compatible experiment tracking, feature store, streaming infrastructure

---

## PM Defense & Quant Research Notes

For insights into the theoretical justifications of this library (why Sharpe Ratio fails in power markets, limitations of deterministic LP battery solvers, defending this architecture in a PM interview), see:

**[docs/PM_DEFENSE.md](docs/PM_DEFENSE.md)**

---

## License

MIT License — Copyright (c) 2026 Gencer Sarp Mert

*Disclaimer: These models are provided for research and educational purposes. Production implementation requires live exchange connectivity, continuous data cleaning, and real-time execution engines with proper risk controls.*
