import numpy as np
import pytest

from energytrading.portfolio.regime_allocation import regime_weighted_kelly, regime_volatility_target
from energytrading.risk.evt import fit_peaks_over_threshold, evt_cvar
from energytrading.models.jump_diffusion import MertonJumpDiffusion


def test_regime_kelly():
    f = regime_weighted_kelly(
        mu_base=0.05, var_base=0.01,
        mu_spike=0.20, var_spike=0.10,
        prob_base=0.9, prob_spike=0.1
    )
    assert f > 0
    assert np.isfinite(f)

def test_regime_vol_target():
    w = regime_volatility_target(0.1, 0.5, 0.9, 0.1, target_vol=0.15)
    assert w > 0

def test_evt_cvar():
    # Generate mock heavy-tailed loss data
    np.random.seed(42)
    returns = -np.random.exponential(scale=0.02, size=1000)
    
    fit = fit_peaks_over_threshold(returns, threshold=0.04)
    cvar = evt_cvar(fit, alpha=0.99)
    assert cvar > 0.04

def test_merton_jump_diffusion():
    mjd = MertonJumpDiffusion(mu=0.0, sigma=0.2, lambda_j=5.0, mu_j=0.1, sigma_j=0.3)
    paths = mjd.simulate(s0=50.0, t_horizon=1.0, dt=1/252, n_paths=2)
    assert paths.shape == (253, 2)
    assert np.all(paths > 0)