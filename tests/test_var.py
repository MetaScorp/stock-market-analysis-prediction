import numpy as np
import pandas as pd
import pytest

from stockanalysis.risk import historical_var, monte_carlo_var, parametric_var


def test_historical_var_matches_manual_quantile():
    returns = pd.Series(np.linspace(-0.05, 0.05, 101))
    result = historical_var(returns, confidence=0.95)
    expected = -np.quantile(returns, 0.05)
    assert result == pytest.approx(expected)


def test_parametric_var_matches_normal_formula():
    from scipy.stats import norm

    returns = pd.Series(np.random.default_rng(0).normal(0.0005, 0.02, 2000))
    result = parametric_var(returns, confidence=0.95)
    expected = -(returns.mean() + norm.ppf(0.05) * returns.std(ddof=1))
    assert result == pytest.approx(expected, rel=1e-6)


def test_monte_carlo_var_is_reproducible_with_same_seed():
    r1 = monte_carlo_var(mu=0.0005, sigma=0.02, n_simulations=5000, seed=42)
    r2 = monte_carlo_var(mu=0.0005, sigma=0.02, n_simulations=5000, seed=42)
    assert r1.var == r2.var
    assert r1.cvar == r2.cvar
    np.testing.assert_array_equal(r1.simulated_returns, r2.simulated_returns)


def test_monte_carlo_var_differs_with_different_seed():
    r1 = monte_carlo_var(mu=0.0005, sigma=0.02, n_simulations=5000, seed=1)
    r2 = monte_carlo_var(mu=0.0005, sigma=0.02, n_simulations=5000, seed=2)
    assert r1.var != r2.var


def test_monte_carlo_cvar_is_at_least_as_severe_as_var():
    result = monte_carlo_var(mu=0.0, sigma=0.02, n_simulations=20_000, seed=42)
    assert result.cvar >= result.var


def test_higher_volatility_increases_var():
    low_vol = monte_carlo_var(mu=0.0, sigma=0.01, n_simulations=20_000, seed=42)
    high_vol = monte_carlo_var(mu=0.0, sigma=0.05, n_simulations=20_000, seed=42)
    assert high_vol.var > low_vol.var
