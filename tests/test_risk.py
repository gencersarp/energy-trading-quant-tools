import numpy as np
from energytrading.risk.metrics import compute_risk_metrics
from energytrading.risk.greeks import compute_delta, compute_gamma

def test_risk_metrics():
    returns = np.array([-0.05, -0.02, 0.0, 0.01, 0.04])
    metrics = compute_risk_metrics(returns, alpha=0.4)
    assert 'VaR' in metrics
    assert metrics['VaR'] == -0.02

def test_greeks():
    # f(x) = x^2 => f'(x) = 2x => f''(x) = 2
    def pricer(x):
        return x**2
        
    delta = compute_delta(pricer, 2.0)
    gamma = compute_gamma(pricer, 2.0)
    
    assert np.isclose(delta, 4.0, atol=1e-3)
    assert np.isclose(gamma, 2.0, atol=1e-3)