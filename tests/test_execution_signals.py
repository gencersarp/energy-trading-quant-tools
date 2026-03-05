"""Tests for execution algorithms and signal modules."""
import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------------ #
# TWAP / VWAP
# ------------------------------------------------------------------ #

class TestExecutionAlgorithms:
    def setup_method(self):
        np.random.seed(42)
        idx = pd.date_range("2024-01-01 09:00", periods=100, freq="5min")
        self.prices = pd.Series(50 + np.random.randn(100).cumsum() * 0.1, index=idx)
        self.volumes = pd.Series(np.random.exponential(1000, 100), index=idx)

    def test_twap_basic(self):
        from energytrading.backtest.twap_vwap import TWAPExecutor
        executor = TWAPExecutor(n_slices=10, add_noise=False)
        result = executor.execute(total_qty=10_000, prices=self.prices)
        assert abs(result.total_qty - 10_000) < 1e-6
        assert result.avg_execution_price > 0
        assert result.algorithm == "TWAP"
        assert len(result.schedule) == 10

    def test_twap_schedule_sums_to_qty(self):
        from energytrading.backtest.twap_vwap import TWAPExecutor
        executor = TWAPExecutor(n_slices=8, add_noise=False)
        result = executor.execute(total_qty=5_000, prices=self.prices)
        assert abs(result.schedule.sum() - 5_000) < 1e-6

    def test_vwap_basic(self):
        from energytrading.backtest.twap_vwap import VWAPExecutor
        executor = VWAPExecutor(n_slices=10, add_noise=False)
        result = executor.execute(total_qty=10_000, prices=self.prices,
                                  volumes=self.volumes)
        assert result.total_qty > 0
        assert result.algorithm == "VWAP"

    def test_is_optimizer(self):
        from energytrading.backtest.twap_vwap import IS_Optimizer
        optimizer = IS_Optimizer(eta=0.1, sigma=0.01, risk_aversion=0.01)
        result = optimizer.execute(total_qty=10_000, prices=self.prices)
        assert result.algorithm == "IS"
        assert result.total_qty > 0

    def test_smart_order_router(self):
        from energytrading.backtest.twap_vwap import SmartOrderRouter
        router = SmartOrderRouter(
            venue_fees={"NYSE": 0.0003, "CBOE": 0.0005},
            venue_impact={"NYSE": 0.001, "CBOE": 0.0008},
        )
        allocation = router.route(
            total_qty=10_000,
            venue_prices={"NYSE": 50.05, "CBOE": 50.08},
            venue_liquidity={"NYSE": 8_000, "CBOE": 5_000},
        )
        total = sum(allocation.values())
        assert abs(total - 10_000) < 1e-6
        assert "NYSE" in allocation or "CBOE" in allocation

    def test_pov_executor(self):
        from energytrading.backtest.twap_vwap import POVExecutor
        executor = POVExecutor(participation_rate=0.1)
        result = executor.execute(total_qty=5_000, prices=self.prices,
                                  volumes=self.volumes)
        assert result.algorithm == "POV"
        assert result.total_qty > 0


# ------------------------------------------------------------------ #
# Signal ensemble
# ------------------------------------------------------------------ #

class TestSignalEnsemble:
    def setup_method(self):
        np.random.seed(7)
        idx = pd.date_range("2023-01-01", periods=200, freq="B")
        self.signals = pd.DataFrame({
            "momentum": np.random.randn(200),
            "mean_rev": np.random.randn(200),
            "ml_signal": np.random.randn(200),
        }, index=idx)
        self.returns = pd.Series(np.random.randn(200) * 0.02, index=idx)

    def test_signal_stack_mean(self):
        from energytrading.signals.ensemble import SignalStack
        stack = SignalStack(method="mean")
        composite = stack.transform(self.signals)
        assert len(composite) == 200
        assert not composite.isna().all()

    def test_signal_stack_weighted(self):
        from energytrading.signals.ensemble import SignalStack
        stack = SignalStack(method="weighted")
        composite = stack.fit_transform(self.signals, self.returns)
        assert len(composite) == 200

    def test_meta_labeler(self):
        from energytrading.signals.ensemble import MetaLabeler
        primary = self.signals["momentum"]
        vol = self.returns.rolling(21).std().fillna(0.01)
        labeler = MetaLabeler(threshold=0.5)
        labeler.fit(primary, self.returns, vol)
        filtered = labeler.filter_signal(primary, self.returns, vol)
        assert len(filtered) == 200
        # Filtered signal should have same or fewer non-zero entries
        assert (filtered != 0).sum() <= (primary != 0).sum()

    def test_bayesian_aggregator(self):
        from energytrading.signals.ensemble import BayesianSignalAggregator
        agg = BayesianSignalAggregator(update_window=30)
        composite = agg.aggregate(self.signals, self.returns)
        assert len(composite) == 200
        assert not composite.isna().all()


# ------------------------------------------------------------------ #
# Calendar spread signals
# ------------------------------------------------------------------ #

class TestCalendarSpreadSignal:
    def setup_method(self):
        np.random.seed(3)
        idx = pd.date_range("2022-01-01", periods=400, freq="B")
        common = np.cumsum(np.random.randn(400)) * 0.5
        self.front = pd.Series(50 + common + np.random.randn(400) * 0.5, index=idx)
        self.back = pd.Series(48 + common + np.random.randn(400) * 0.3, index=idx)

    def test_compute_spread(self):
        from energytrading.signals.calendar_spread import CalendarSpreadSignal
        gen = CalendarSpreadSignal()
        spread = gen.compute_spread(self.front, self.back)
        assert len(spread) == 400
        assert spread.mean() > 0  # front typically higher

    def test_signal_values(self):
        from energytrading.signals.calendar_spread import CalendarSpreadSignal
        gen = CalendarSpreadSignal(entry_z=1.5, lookback=50)
        signal = gen.compute_signal(self.front, self.back)
        assert set(signal.unique()).issubset({-1.0, 0.0, 1.0})

    def test_rolling_stats(self):
        from energytrading.signals.calendar_spread import CalendarSpreadSignal
        gen = CalendarSpreadSignal()
        stats = gen.rolling_stats(self.front, self.back)
        assert "z_score" in stats.columns
        assert "spread" in stats.columns

    def test_vol_arb_signal(self):
        from energytrading.signals.calendar_spread import VolatilityArbitrageSignal
        gen = VolatilityArbitrageSignal(threshold=1.2)
        signal = gen.compute_signal(self.front)
        assert set(signal.unique()).issubset({-1.0, 0.0, 1.0})

    def test_cross_commodity_signal(self):
        from energytrading.signals.calendar_spread import CrossCommoditySpreadSignal
        gen = CrossCommoditySpreadSignal(entry_z=1.5, lookback=50)
        signal, spread = gen.compute_signal(self.front, self.back)
        assert len(signal) == len(self.front)
        assert len(spread) == len(self.front)


# ------------------------------------------------------------------ #
# Microstructure
# ------------------------------------------------------------------ #

class TestMarketMicrostructure:
    def test_order_book_basic(self):
        from energytrading.market_microstructure.order_book import OrderBook, Order
        import time as _time
        book = OrderBook()
        bid = Order(order_id="B1", side="bid", price=50.0, quantity=100,
                    timestamp=_time.time())
        ask = Order(order_id="A1", side="ask", price=50.5, quantity=100,
                    timestamp=_time.time())
        book.add_order(bid)
        book.add_order(ask)
        assert book.best_bid() == 50.0
        assert book.best_ask() == 50.5

    def test_order_book_match(self):
        from energytrading.market_microstructure.order_book import OrderBook, Order
        import time as _time
        book = OrderBook()
        bid = Order("B1", "bid", 51.0, 100, timestamp=_time.time())
        ask = Order("A1", "ask", 50.0, 100, timestamp=_time.time() + 0.001)
        book.add_order(bid)
        fills = book.add_order(ask)
        assert len(fills) > 0
        assert fills[0].price == 51.0  # price-time priority

    def test_impact_models(self):
        from energytrading.market_microstructure.impact import (
            LinearImpactModel, SquareRootImpactModel)
        lm = LinearImpactModel()
        lm._eta = 0.1
        impact = lm.estimate(volume=1000, adv=100_000)
        assert impact > 0

        sqm = SquareRootImpactModel()
        sqm._gamma = 1.0
        impact = sqm.estimate(volume=1000, adv=100_000, sigma=0.02)
        assert impact > 0

    def test_almgren_chriss(self):
        from energytrading.market_microstructure.impact import AlmgrenChrissOptimizer
        opt = AlmgrenChrissOptimizer(S0=50, V=10_000, sigma=0.02,
                                     eta=0.05, gamma=0.01, tau=1.0)
        schedule = opt.optimal_trajectory(n_steps=10, risk_aversion=0.01)
        assert len(schedule) == 10
        assert abs(schedule.sum() - 10_000) < 100  # close to total volume
        assert np.all(schedule >= 0)
