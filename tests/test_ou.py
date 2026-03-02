import numpy as np

from energytrading.models.ou import OUParams, fit_ou_mle, simulate_ou


def test_simulate_shape_and_finite():
    params = OUParams(mu=10.0, theta=1.0, sigma=2.0, dt=1 / 252)
    x = simulate_ou(x0=10.0, n_steps=1000, params=params, seed=123)
    assert x.shape == (1001,)
    assert np.all(np.isfinite(x))


def test_fit_recovers_reasonable_parameters():
    true = OUParams(mu=50.0, theta=1.5, sigma=5.0, dt=1.0)
    x = simulate_ou(x0=20.0, n_steps=4000, params=true, seed=7)
    fit = fit_ou_mle(x, dt=1.0)

    # Loose tolerances; stochastic.
    assert abs(fit.params.mu - true.mu) < 2.0
    assert abs(fit.params.theta - true.theta) < 0.25
    assert abs(fit.params.sigma - true.sigma) < 1.0