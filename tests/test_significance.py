import numpy as np
import pandas as pd

from stockanalysis.backtest import sma_crossover_signal
from stockanalysis.evaluation import bootstrap_sharpe_ci, permutation_test_signal


def _dates(n):
    return pd.bdate_range("2020-01-01", periods=n)


def test_bootstrap_point_estimate_matches_plain_sharpe():
    from stockanalysis.stats import sharpe_ratio

    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 500), index=_dates(500))
    result = bootstrap_sharpe_ci(returns, n_bootstrap=300)
    assert result.point_estimate == sharpe_ratio(returns)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 500), index=_dates(500))
    result = bootstrap_sharpe_ci(returns, n_bootstrap=300)
    assert result.lower <= result.point_estimate <= result.upper


def test_bootstrap_ci_widens_with_less_data():
    rng = np.random.default_rng(1)
    long_returns = pd.Series(rng.normal(0.0005, 0.01, 1000), index=_dates(1000))
    short_returns = long_returns.iloc[:60]

    long_ci = bootstrap_sharpe_ci(long_returns, n_bootstrap=500)
    short_ci = bootstrap_sharpe_ci(short_returns, n_bootstrap=500)

    assert (short_ci.upper - short_ci.lower) > (long_ci.upper - long_ci.lower)


def test_bootstrap_is_reproducible_with_same_seed():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.001, 0.01, 300), index=_dates(300))
    r1 = bootstrap_sharpe_ci(returns, n_bootstrap=200, seed=7)
    r2 = bootstrap_sharpe_ci(returns, n_bootstrap=200, seed=7)
    np.testing.assert_array_equal(r1.samples, r2.samples)


def test_permutation_p_value_is_between_zero_and_one():
    rng = np.random.default_rng(3)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=_dates(300))
    returns = prices.pct_change().fillna(0)
    signal = sma_crossover_signal(prices, fast=10, slow=30)

    result = permutation_test_signal(signal, returns, n_permutations=150)
    assert 0.0 <= result.p_value <= 1.0
    assert len(result.null_sharpes) == 150


def test_permutation_test_flags_an_engineered_edge_as_significant():
    # build a signal that's only long right before an engineered upward jump,
    # so it should clearly beat a random selection of the same number of days
    rng = np.random.default_rng(4)
    n = 400
    returns = pd.Series(rng.normal(0, 0.005, n), index=_dates(n))
    jump_days = np.arange(20, n, 20)
    returns.iloc[jump_days] += 0.05  # big known jumps

    signal = pd.Series(0, index=returns.index)
    signal.iloc[jump_days - 1] = 1  # signal fires the day before every jump

    result = permutation_test_signal(signal, returns, n_permutations=300)
    assert result.p_value < 0.05
