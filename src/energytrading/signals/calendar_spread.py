"""Calendar spread, seasonal spread, and volatility arbitrage signals."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class CalendarSpreadSignal:
    """
    Trading signal based on calendar (time) spread between front and back months.
    Captures contango/backwardation dynamics in energy futures.
    """

    def __init__(self, entry_z: float = 1.5, exit_z: float = 0.3,
                 lookback: int = 63):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback

    def compute_spread(self, front: pd.Series, back: pd.Series) -> pd.Series:
        """Calendar spread = front - back (positive = backwardation)."""
        return front - back

    def compute_signal(self, front: pd.Series, back: pd.Series,
                       side: str = "both") -> pd.Series:
        """
        Mean-reversion signal on calendar spread z-score.
        side: 'long', 'short', or 'both'
        Returns: position in [-1, +1]
        """
        spread = self.compute_spread(front, back)
        roll = spread.rolling(self.lookback, min_periods=10)
        z = (spread - roll.mean()) / (roll.std() + 1e-8)

        signal = pd.Series(0.0, index=spread.index)
        if side in ("both", "long"):
            signal[z < -self.entry_z] = 1.0    # spread too low → buy front
        if side in ("both", "short"):
            signal[z > self.entry_z] = -1.0    # spread too high → sell front

        # Exit at mean reversion
        signal[(signal == 0) & (z.abs() < self.exit_z)] = 0.0
        return signal.rename("calendar_spread_signal")

    def rolling_stats(self, front: pd.Series,
                       back: pd.Series) -> pd.DataFrame:
        spread = self.compute_spread(front, back)
        roll = spread.rolling(self.lookback, min_periods=5)
        z = (spread - roll.mean()) / (roll.std() + 1e-8)
        return pd.DataFrame({
            "spread": spread,
            "spread_mean": roll.mean(),
            "spread_std": roll.std(),
            "z_score": z,
            "percentile": z.rank(pct=True),
        })

    def curve_structure(self, curves: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Analyze the term structure across multiple tenors.
        curves: {'M1': series_M1, 'M2': series_M2, 'M3': ..., 'Q1': ...}
        """
        tenors = sorted(curves.keys())
        result = pd.DataFrame(curves)
        # Compute all adjacent spreads
        for i in range(len(tenors) - 1):
            t1, t2 = tenors[i], tenors[i + 1]
            result[f"spread_{t1}_{t2}"] = curves[t1] - curves[t2]
        return result


class SeasonalSpreadSignal:
    """
    Signal based on seasonal price differentials:
    summer/winter gas, on/off-peak power, heating/cooling.
    """

    def __init__(self, entry_z: float = 1.5, lookback_years: int = 3):
        self.entry_z = entry_z
        self.lookback_years = lookback_years

    def seasonal_premium(self, price: pd.Series,
                          high_season_months: List[int],
                          low_season_months: List[int]) -> pd.Series:
        """Compute rolling seasonal premium (high_season - low_season avg)."""
        high = price[price.index.month.isin(high_season_months)]
        low = price[price.index.month.isin(low_season_months)]

        # Rolling annual average of each season
        result = pd.Series(np.nan, index=price.index)
        window = f"{self.lookback_years * 365}D"

        for date in price.index:
            period_start = date - pd.DateOffset(years=self.lookback_years)
            h = high[(high.index >= period_start) & (high.index <= date)]
            l = low[(low.index >= period_start) & (low.index <= date)]
            if len(h) > 5 and len(l) > 5:
                result[date] = h.mean() - l.mean()

        return result.rename("seasonal_premium")

    def winter_summer_signal(self, winter_price: pd.Series,
                              summer_price: pd.Series) -> pd.Series:
        """Signal on winter/summer spread (natural gas)."""
        spread = winter_price - summer_price
        z = (spread - spread.rolling(252, min_periods=30).mean()) / \
            (spread.rolling(252, min_periods=30).std() + 1e-8)

        signal = pd.Series(0.0, index=spread.index)
        signal[z > self.entry_z] = 1.0    # winter expensive → long winter/short summer
        signal[z < -self.entry_z] = -1.0  # summer expensive → opposite
        return signal.rename("winter_summer_signal")

    def day_of_week_signal(self, price: pd.Series,
                            lookback: int = 52) -> pd.DataFrame:
        """Identify day-of-week seasonal patterns in electricity prices."""
        df = pd.DataFrame({"price": price, "dow": price.index.dayofweek})
        dow_means = {}
        for dow in range(7):
            series = df[df["dow"] == dow]["price"]
            rolling_mean = series.rolling(lookback, min_periods=10).mean()
            dow_means[dow] = rolling_mean

        result = pd.DataFrame(dow_means, index=price.index)
        return result


class VolatilityArbitrageSignal:
    """
    Signal based on realized vs implied volatility spread.
    When RV < IV: sell vol (short gamma). When RV > IV: buy vol.
    """

    def __init__(self, rv_window: int = 21, iv_proxy_window: int = 63,
                 threshold: float = 1.2):
        self.rv_window = rv_window
        self.iv_proxy_window = iv_proxy_window
        self.threshold = threshold  # RV/IV ratio trigger

    def realized_vol(self, price: pd.Series) -> pd.Series:
        log_ret = np.log(price / price.shift(1))
        return log_ret.rolling(self.rv_window).std() * np.sqrt(252)

    def implied_vol_proxy(self, price: pd.Series,
                           option_price: Optional[pd.Series] = None) -> pd.Series:
        """
        Proxy for implied vol using longer-window historical vol or
        provided option price series.
        """
        if option_price is not None:
            # Simple Black-Scholes IV approximation (ATM straddle / price)
            return option_price / (price + 1e-8) * np.sqrt(252 / self.rv_window)
        return np.log(price / price.shift(1)).rolling(self.iv_proxy_window).std() * np.sqrt(252)

    def vol_spread(self, price: pd.Series,
                    option_price: Optional[pd.Series] = None) -> pd.Series:
        rv = self.realized_vol(price)
        iv = self.implied_vol_proxy(price, option_price)
        return rv - iv

    def compute_signal(self, price: pd.Series,
                        option_price: Optional[pd.Series] = None) -> pd.Series:
        """
        +1 = buy vol (RV > IV, vol is cheap)
        -1 = sell vol (IV > RV, vol is rich)
         0 = no trade
        """
        rv = self.realized_vol(price)
        iv = self.implied_vol_proxy(price, option_price)
        ratio = rv / (iv + 1e-8)

        signal = pd.Series(0.0, index=price.index)
        signal[ratio > self.threshold] = 1.0    # buy realized vol
        signal[ratio < 1.0 / self.threshold] = -1.0  # sell realized vol
        return signal.rename("vol_arb_signal")

    def vol_term_structure_signal(self, short_vol: pd.Series,
                                   long_vol: pd.Series) -> pd.Series:
        """
        Trade the term structure of volatility:
        long short-term vol when vol curve is inverted.
        """
        spread = short_vol - long_vol
        z = (spread - spread.rolling(63, min_periods=10).mean()) / \
            (spread.rolling(63, min_periods=10).std() + 1e-8)

        signal = pd.Series(0.0, index=spread.index)
        signal[z > 1.5] = 1.0   # short-vol expensive → vol curve steep (sell short)
        signal[z < -1.5] = -1.0  # inverted vol curve (buy short-vol)
        return signal.rename("vol_term_structure_signal")


class CrossCommoditySpreadSignal:
    """
    Statistical arbitrage across commodity pairs:
    crude/products crack, gas/power spark, coal/power dark.
    """

    def __init__(self, lookback: int = 63, entry_z: float = 2.0,
                 exit_z: float = 0.5):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self._hedge_ratio: float = 1.0

    def estimate_hedge_ratio(self, commodity1: pd.Series,
                              commodity2: pd.Series) -> float:
        """OLS-based hedge ratio estimation."""
        x = commodity2.values
        y = commodity1.values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 5:
            return 1.0
        x = x[mask]
        y = y[mask]
        self._hedge_ratio = float(np.dot(x, y) / (np.dot(x, x) + 1e-8))
        return self._hedge_ratio

    def compute_signal(self, commodity1: pd.Series,
                        commodity2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Returns (signal, spread) where signal is mean-reversion direction.
        """
        beta = self.estimate_hedge_ratio(commodity1, commodity2)
        spread = commodity1 - beta * commodity2
        roll = spread.rolling(self.lookback, min_periods=10)
        z = (spread - roll.mean()) / (roll.std() + 1e-8)

        signal = pd.Series(0.0, index=spread.index)
        signal[z < -self.entry_z] = 1.0    # spread low → long C1/short C2
        signal[z > self.entry_z] = -1.0    # spread high → short C1/long C2
        # Exit on reversion
        in_trade = signal != 0
        signal[~in_trade & (z.abs() < self.exit_z)] = 0.0

        return signal.rename("cross_commodity_signal"), spread.rename("spread")
