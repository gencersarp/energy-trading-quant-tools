from energytrading.signals.microstructure import order_book_imbalance, calculate_vwap
from energytrading.pricing.imbalance import dual_pricing_cashout
from energytrading.signals.renewables import renewable_ramp_signal
import pandas as pd
import numpy as np

def test_order_book_imbalance():
    bid = pd.Series([100, 50, 0])
    ask = pd.Series([50, 100, 10])
    obi = order_book_imbalance(bid, ask)
    assert np.isclose(obi.iloc[0], 0.333, atol=0.01) # (100-50)/150 = 50/150 = 0.333
    assert np.isclose(obi.iloc[1], -0.333, atol=0.01) # (50-100)/150 = -0.333
    assert obi.iloc[2] == -1.0 # (0-10)/10 = -1.0

def test_vwap():
    prices = np.array([10.0, 12.0])
    volumes = np.array([100, 200])
    v = calculate_vwap(prices, volumes)
    assert np.isclose(v, 11.333, atol=0.01) # (1000 + 2400) / 300 = 11.333

def test_dual_pricing_cashout():
    # Long power (we bought it). Grid is LONG (too much power). We face a penalty.
    pnl = dual_pricing_cashout(position_mwh=10.0, market_price=50.0, system_state="LONG", penalty_markup=0.2)
    assert pnl == 400.0 # 10 * 50 * 0.8 (we lose 20% value dumping it)
    
    # Short power. Grid is SHORT. We face a penalty.
    pnl_short = dual_pricing_cashout(position_mwh=-10.0, market_price=50.0, system_state="SHORT", penalty_markup=0.2)
    assert pnl_short == -600.0 # -10 * 50 * 1.2 (we have to buy it back 20% higher)

def test_renewable_ramp():
    solar = pd.Series([5000, 4500, 2000, 2000])
    wind = pd.Series([5000, 5000, 5000, 7000])
    # T0: 10000 -> T1: 9500 -> T2: 7000 (drop of 2500) -> T3: 9000 (surge of 2000)
    sig = renewable_ramp_signal(solar, wind, ramp_threshold_mw=1500)
    
    assert sig.iloc[1] == 0
    assert sig.iloc[2] == 1 # Drop of 2500 triggered Buy
    assert sig.iloc[3] == -1 # Surge of 2000 triggered Sell