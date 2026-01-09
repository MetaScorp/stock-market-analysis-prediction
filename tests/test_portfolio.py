import numpy as np
import pandas as pd
import pytest

from stockanalysis.portfolio import (
    beta,
    relative_strength,
    rolling_beta,
    rolling_correlation,
)


def _dates(n):
    return pd.bdate_range("2022-01-03", periods=n)


def test_rolling_correlation_is_high_for_near_identical_series():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 0.01, 200), index=_dates(200))
    noisy_copy = base + pd.Series(rng.normal(0, 0.0001, 200), index=_dates(200))
    result = rolling_correlation(base, noisy_copy, window=30).dropna()
    assert (result > 0.9).all()


def test_rolling_correlation_near_zero_for_independent_series():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0, 0.01, 500), index=_dates(500))
    b = pd.Series(rng.normal(0, 0.01, 500), index=_dates(500))
    result = rolling_correlation(a, b, window=60).dropna()
    assert abs(result.mean()) < 0.15


def test_beta_of_one_for_identical_returns():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0, 0.01, 300), index=_dates(300))
    assert beta(returns, returns) == pytest.approx(1.0)


def test_beta_of_two_for_scaled_returns():
    rng = np.random.default_rng(3)
    market = pd.Series(rng.normal(0, 0.01, 300), index=_dates(300))
    leveraged = market * 2
    assert beta(leveraged, market) == pytest.approx(2.0)


def test_rolling_beta_matches_static_beta_on_stationary_series():
    rng = np.random.default_rng(4)
    market = pd.Series(rng.normal(0, 0.01, 400), index=_dates(400))
    asset = market * 1.5 + rng.normal(0, 0.001, 400)
    result = rolling_beta(asset, market, window=100).dropna()
    assert result.mean() == pytest.approx(1.5, abs=0.1)


def test_relative_strength_starts_at_one():
    a = pd.Series([100.0, 110.0, 120.0], index=_dates(3))
    b = pd.Series([50.0, 51.0, 49.0], index=_dates(3))
    result = relative_strength(a, b)
    assert result.iloc[0] == pytest.approx(1.0)


def test_relative_strength_rises_when_asset_outperforms():
    a = pd.Series([100.0, 110.0, 130.0], index=_dates(3))
    b = pd.Series([100.0, 100.0, 100.0], index=_dates(3))
    result = relative_strength(a, b)
    assert result.is_monotonic_increasing
