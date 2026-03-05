"""Weather derivatives pricing: HDD/CDD options, swaps, burn analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class DegreeDayOption:
    option_type: str   # 'call' | 'put' (call on HDD/CDD index)
    strike_dd: float   # strike in degree-days
    tick_size: float   # EUR per degree-day
    period_start: str
    period_end: str
    cap: Optional[float] = None  # maximum payout


class WeatherDerivativesPricer:
    """Price weather derivatives using burn analysis and parametric methods."""

    def _simulate_dd_index(self, temp_scenarios: np.ndarray,
                            dd_type: str = "HDD",
                            base_temp: float = 18.0) -> np.ndarray:
        """Compute HDD/CDD index for each path."""
        if dd_type == "HDD":
            dd = np.maximum(base_temp - temp_scenarios, 0)
        else:
            dd = np.maximum(temp_scenarios - base_temp, 0)
        return dd.sum(axis=-1) if dd.ndim > 1 else np.array([dd.sum()])

    def price_hdd_option(self, option: DegreeDayOption,
                          temp_scenarios: np.ndarray,
                          base_temp: float = 18.0) -> Tuple[float, float, float]:
        """Price HDD option via Monte Carlo simulation."""
        dd_index = self._simulate_dd_index(temp_scenarios, "HDD", base_temp)
        if option.option_type == "call":
            payoffs = np.maximum(dd_index - option.strike_dd, 0) * option.tick_size
        else:
            payoffs = np.maximum(option.strike_dd - dd_index, 0) * option.tick_size
        if option.cap is not None:
            payoffs = np.minimum(payoffs, option.cap)
        price = float(payoffs.mean())
        delta = float(np.mean(dd_index > option.strike_dd)
                      if option.option_type == "call"
                      else np.mean(dd_index < option.strike_dd))
        gamma = float(np.std(payoffs) / (dd_index.std() * option.tick_size + 1e-8))
        return price, delta, gamma

    def price_cdd_option(self, option: DegreeDayOption,
                          temp_scenarios: np.ndarray,
                          base_temp: float = 18.0) -> Tuple[float, float, float]:
        return self.price_hdd_option(option, temp_scenarios, base_temp)

    def price_hdd_swap(self, strike: float, tick_size: float,
                        period: str, temp_forecast: pd.Series,
                        vol: float = 5.0) -> float:
        """Fair value of HDD swap: expected HDD vs strike."""
        daily_temp = temp_forecast.resample("D").mean() if hasattr(temp_forecast.index, "freq") else temp_forecast
        expected_hdd = float(np.maximum(18.0 - daily_temp, 0).sum())
        n_days = len(daily_temp)
        # Fair value = (E[HDD] - strike) * tick_size
        return float((expected_hdd - strike) * tick_size)

    def burn_analysis(self, option: DegreeDayOption,
                       historical_temps: pd.DataFrame,
                       base_temp: float = 18.0) -> float:
        """Historical simulation (burn analysis) price."""
        payoffs = []
        for col in historical_temps.columns:
            temps = historical_temps[col].values
            dd = np.maximum(base_temp - temps, 0).sum() if "H" in option.option_type.upper() else np.maximum(temps - base_temp, 0).sum()
            if option.option_type == "call":
                payout = max(dd - option.strike_dd, 0) * option.tick_size
            else:
                payout = max(option.strike_dd - dd, 0) * option.tick_size
            if option.cap is not None:
                payout = min(payout, option.cap)
            payoffs.append(payout)
        return float(np.mean(payoffs))

    def implied_vol_from_price(self, option: DegreeDayOption,
                                market_price: float,
                                temp_scenarios: np.ndarray,
                                base_temp: float = 18.0) -> float:
        """Newton's method to find implied temperature volatility."""
        dd_index = self._simulate_dd_index(temp_scenarios, "HDD", base_temp)
        mu_dd = float(dd_index.mean())
        # Binary search over vol
        lo, hi = 0.01, 100.0
        for _ in range(50):
            mid_vol = (lo + hi) / 2
            # Price with mid_vol (scale scenarios)
            scaled = dd_index * mid_vol / (dd_index.std() + 1e-8)
            if option.option_type == "call":
                payoffs = np.maximum(scaled + mu_dd * (1 - mid_vol / (dd_index.std() + 1e-8)) - option.strike_dd, 0) * option.tick_size
            else:
                payoffs = np.maximum(option.strike_dd - scaled, 0) * option.tick_size
            model_price = float(payoffs.mean())
            if model_price < market_price:
                lo = mid_vol
            else:
                hi = mid_vol
        return float((lo + hi) / 2)

    def index_correlation(self, dd_index: pd.Series,
                           energy_prices: pd.Series) -> float:
        """Pearson correlation between DD index and energy prices for cross-hedging."""
        aligned = pd.concat([dd_index, energy_prices], axis=1).dropna()
        if len(aligned) < 3:
            return 0.0
        return float(aligned.corr().iloc[0, 1])
