import numpy as np
import pandas as pd
import pytest

from stockanalysis.stats import (
    annualized_return,
    annualized_volatility,
    correlation_matrix,
    cumulative_returns,
    daily_returns,
    sharpe_ratio,
)


def test_daily_returns_simple():
    prices = pd.Series([100.0, 110.0, 99.0])
    result = daily_returns(prices)
    assert result.iloc[0] == pytest.approx(0.10)
    assert result.iloc[1] == pytest.approx(-0.10)


def test_daily_returns_log():
    prices = pd.Series([100.0, 110.0])
    result = daily_returns(prices, log=True)
    assert result.iloc[0] == pytest.approx(np.log(1.10))


def test_cumulative_returns_compounds():
    returns = pd.Series([0.10, -0.10])
    result = cumulative_returns(returns)
    assert result.iloc[-1] == pytest.approx(1.10 * 0.90)


def test_annualized_return_of_constant_daily_return():
    returns = pd.Series([0.001] * 252)
    result = annualized_return(returns, periods_per_year=252)
    assert result == pytest.approx(1.001**252 - 1)


def test_annualized_volatility_scales_with_sqrt_time():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 1000))
    result = annualized_volatility(returns, periods_per_year=252)
    assert result == pytest.approx(returns.std(ddof=1) * np.sqrt(252))


def test_sharpe_ratio_zero_for_zero_excess_return():
    returns = pd.Series([0.0] * 10)
    assert np.isnan(sharpe_ratio(returns))  # zero variance -> undefined


def test_sharpe_ratio_positive_for_positive_mean_return():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    assert sharpe_ratio(returns) > 0


def test_correlation_matrix_diagonal_is_one():
    returns_a = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
    returns_b = -returns_a
    prices = pd.DataFrame(
        {
            "A": 100 * (1 + returns_a).cumprod(),
            "B": 100 * (1 + returns_b).cumprod(),
        }
    )
    corr = correlation_matrix(prices)
    assert corr.loc["A", "A"] == pytest.approx(1.0)
    assert corr.loc["A", "B"] == pytest.approx(-1.0)
