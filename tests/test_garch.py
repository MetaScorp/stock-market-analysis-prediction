import numpy as np
import pandas as pd
import pytest

from stockanalysis.risk.garch import fit_garch, garch_var
from stockanalysis.risk.var import parametric_var


def _returns_with_vol_clustering(seed=0, n=1500):
    """Simulate returns where volatility clusters (a real GARCH(1,1) process),
    so there's something for the model to actually pick up on."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 1e-6, 0.1, 0.85
    sigma2 = np.empty(n)
    returns = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)
    returns[0] = rng.normal(0, np.sqrt(sigma2[0]))
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))
    return pd.Series(returns)


def test_fit_garch_returns_positive_forecast_volatility():
    returns = _returns_with_vol_clustering()
    result = fit_garch(returns, horizon_days=5)
    assert len(result.forecast_volatility) == 5
    assert (result.forecast_volatility > 0).all()


def test_fit_garch_conditional_volatility_matches_series_length():
    returns = _returns_with_vol_clustering()
    result = fit_garch(returns)
    assert len(result.conditional_volatility) == len(returns.dropna())


def test_garch_conditional_vol_is_higher_during_the_turbulent_stretch():
    rng = np.random.default_rng(5)
    calm = rng.normal(0, 0.003, 400)
    turbulent = rng.normal(0, 0.04, 400)
    returns = pd.Series(np.concatenate([calm, turbulent]))
    result = fit_garch(returns)
    calm_avg = result.conditional_volatility.iloc[100:300].mean()
    turbulent_avg = result.conditional_volatility.iloc[500:700].mean()
    assert turbulent_avg > calm_avg


def test_annualized_forecast_scales_by_sqrt_252():
    returns = _returns_with_vol_clustering()
    result = fit_garch(returns, horizon_days=1)
    annualized = result.annualized_forecast(periods_per_year=252)
    assert annualized[0] == pytest.approx(result.forecast_volatility[0] * 252 ** 0.5)


def test_garch_var_is_positive_and_in_a_plausible_range():
    returns = _returns_with_vol_clustering()
    result = garch_var(returns, confidence=0.95)
    assert 0 < result < 1


def test_garch_var_differs_from_flat_parametric_var_when_vol_is_clustered():
    returns = _returns_with_vol_clustering()
    garch_result = garch_var(returns, confidence=0.95)
    flat_result = parametric_var(returns, confidence=0.95)
    # not asserting a direction - just that GARCH actually uses different
    # information than a single full-sample std dev
    assert garch_result != pytest.approx(flat_result, rel=1e-3)
