import numpy as np
import pandas as pd
from energytrading.risk.metrics import compute_risk_metrics, compute_parametric_var
from energytrading.risk.greeks import compute_delta, compute_gamma
from energytrading.backtest.engine import BacktestEngine, Strategy
from energytrading.models.schwartz_smith import SchwartzSmithModel

def test_risk_metrics():
    returns = np.array([-0.05, -0.02, 0.0, 0.01, 0.04])
    metrics = compute_risk_metrics(returns, alpha=0.4)
    assert 'Historical_VaR' in metrics
    # At alpha 0.4 on 5 elements, idx is 2, value is 0.0
    assert metrics['Historical_VaR'] == 0.0

def test_parametric_var():
    returns = np.random.normal(0, 0.01, 1000)
    p = compute_parametric_var(returns)
    assert 'Parametric_VaR' in p
    assert 'Parametric_CVaR' in p

def test_greeks():
    def pricer(x):
        return x**2
        
    delta = compute_delta(pricer, 2.0)
    gamma = compute_gamma(pricer, 2.0)
    
    assert np.isclose(delta, 4.0, atol=1e-3)
    assert np.isclose(gamma, 2.0, atol=1e-3)

class DummyStrategy(Strategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=data.index)

def test_backtest_metrics():
    df = pd.DataFrame({'close': [100, 101, 102, 99, 105]})
    engine = BacktestEngine(df, DummyStrategy(), transaction_cost=0.001)
    res = engine.run()
    metrics = engine.get_metrics(res)
    assert metrics['Total_Return'] > -1.0 # returns valid float
    assert metrics['Max_Drawdown'] <= 0.0
    assert 'Sharpe_Ratio' in metrics

def test_schwartz_smith():
    model = SchwartzSmithModel()
    prices = np.array([50, 52, 49, 51, 55])
    model.fit_kalman(prices)
    sims = model.simulate(50.0, 10, n_paths=2)
    assert sims.shape == (11, 2)