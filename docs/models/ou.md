# Ornstein-Uhlenbeck (OU)

The OU process is a classic mean-reverting model used as a baseline for energy spot prices and spreads.

## Parameters

- `mu`: long-run mean
- `theta`: reversion speed
- `sigma`: volatility
- `dt`: observation / simulation step

## Practical notes for energy

- Power often needs a spike component (jumps or regimes) layered on top of mean reversion.
- Seasonality is usually essential (hour-of-day, day-of-week, monthly effects).

## API

- `simulate_ou(x0, n_steps, params, seed=None)`
- `fit_ou_mle(x, dt=1.0)`