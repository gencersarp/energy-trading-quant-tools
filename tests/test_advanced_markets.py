import numpy as np
import pandas as pd

from energytrading.models.spatial_copula import SpatialWindCopula
from energytrading.pricing.spreads import calculate_clean_dark_spread, calculate_clean_spark_spread
from energytrading.strategy.virtual_bidding import moving_average_convergence_signal, virtual_bidding_pnl


def test_spark_spread():
    power = pd.Series([100.0, 150.0])
    gas = pd.Series([20.0, 25.0])
    carbon = pd.Series([80.0, 80.0])
    
    # Gas cost = 20 / 0.5 = 40. Carbon cost = 80 * (0.2/0.5) = 32. 
    # CSS = 100 - 40 - 32 = 28
    css = calculate_clean_spark_spread(power, gas, carbon, efficiency=0.5, emission_factor=0.2)
    assert np.isclose(css.iloc[0], 28.0)


def test_dark_spread():
    power = pd.Series([100.0, 150.0])
    coal = pd.Series([20.0, 25.0])
    carbon = pd.Series([80.0, 80.0])
    
    cds = calculate_clean_dark_spread(power, coal, carbon, efficiency=0.35, emission_factor=0.34)
    assert len(cds) == 2


def test_spatial_copula():
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    copula = SpatialWindCopula(correlation_matrix=corr, shape_params=[2.0, 2.0], scale_params=[8.0, 9.0])
    
    np.random.seed(42)
    winds = copula.simulate_wind_speeds(1000)
    
    assert winds.shape == (1000, 2)
    # Check that empirical correlation is highly positive
    emp_corr = np.corrcoef(winds[:, 0], winds[:, 1])[0, 1]
    assert emp_corr > 0.6  # Rank correlation preserves strong positive correlation


def test_virtual_bidding():
    da = pd.Series([50.0, 60.0, 40.0])
    rt = pd.Series([40.0, 70.0, 40.0])
    bids = pd.Series([1, -1, 0]) # INC, DEC, None
    vol = pd.Series([10.0, 10.0, 10.0])
    
    pnl = virtual_bidding_pnl(da, rt, bids, vol, transaction_fee=1.0)
    
    # T0 (INC): Sell DA @ 50, Buy RT @ 40 => +10 profit * 10MW = 100. Minus 10 fee = 90
    assert pnl.iloc[0] == 90.0
    # T1 (DEC): Buy DA @ 60, Sell RT @ 70 => +10 profit * 10MW = 100. Minus 10 fee = 90
    assert pnl.iloc[1] == 90.0
    # T2 (None): 0 profit
    assert pnl.iloc[2] == 0.0


def test_moving_average_convergence_signal():
    da = pd.Series([50.0, 60.0, 40.0, 50.0, 60.0])
    rt = pd.Series([40.0, 70.0, 40.0, 55.0, 65.0])
    
    signal = moving_average_convergence_signal(da, rt, window=2)
    assert len(signal) == 5