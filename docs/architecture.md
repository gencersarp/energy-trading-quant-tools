# System Architecture

## 1. Models (`src/energytrading/models/`)
Stochastic differential equations including Ornstein-Uhlenbeck and Merton Jump-Diffusion models for Monte Carlo pricing simulations.

## 2. Pricing (`src/energytrading/pricing/`)
Options and real options frameworks, including Margrabe Spark Spread option pricing.

## 3. Risk (`src/energytrading/risk/`)
Calculates Historical Value at Risk (VaR), Conditional VaR (Expected Shortfall).

## 4. Backtesting (`src/energytrading/backtest/`)
Vectorized backtesting engine utilizing pandas protocols for structural strategies.

## 5. Streaming (`src/energytrading/data/`)
Kafka consumer/producer wrappers for real-time tick ingestion (stubs for ICE/EEX).

## 6. API (`src/energytrading/api/`)
FastAPI wrapping the quantitative models for external REST consumption.

## 7. Infrastructure
Dockerized environments, `docker-compose` with Zookeeper/Kafka, and Kubernetes (K8s) deployment manifests supporting GPU pass-through.