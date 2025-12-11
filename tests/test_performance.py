import numpy as np
import pandas as pd
import pytest

from stockanalysis.stats import (
    calmar_ratio,
    drawdown_series,
    max_drawdown_duration,
    profit_factor,
    sortino_ratio,
    win_rate,
)


def test_drawdown_series_zero_at_new_highs():
    equity = pd.Series([100.0, 110.0, 105.0, 120.0])
    dd = drawdown_series(equity)
    assert dd.iloc[0] == pytest.approx(0.0)
    assert dd.iloc[1] == pytest.approx(0.0)
    assert dd.iloc[2] == pytest.approx(105 / 110 - 1)
    assert dd.iloc[3] == pytest.approx(0.0)


def test_max_drawdown_duration_counts_longest_underwater_stretch():
    equity = pd.Series([100.0, 90.0, 95.0, 99.0, 101.0, 100.0])
    # underwater for indices 1,2,3 (3 bars) before a new high at index 4
    assert max_drawdown_duration(equity) == 3


def test_max_drawdown_duration_zero_for_monotonic_series():
    equity = pd.Series([100.0, 101.0, 102.0, 103.0])
    assert max_drawdown_duration(equity) == 0


def test_sortino_ignores_upside_volatility():
    # same downside days in both series (nonzero variance), only the
    # upside days differ in size
    calm_upside = pd.Series([0.001, 0.05, -0.01, 0.06, -0.02])
    volatile_upside = pd.Series([0.001, 0.20, -0.01, 0.25, -0.02])
    assert sortino_ratio(volatile_upside) > sortino_ratio(calm_upside)
    # ...because the mean excess return is higher while downside risk is the same


def test_sortino_nan_when_no_downside():
    returns = pd.Series([0.01, 0.02, 0.03])
    assert np.isnan(sortino_ratio(returns))


def test_calmar_ratio_positive_for_profitable_low_drawdown_strategy():
    returns = pd.Series([0.001] * 100 + [-0.01] + [0.001] * 100)
    equity = 100 * (1 + returns).cumprod()
    result = calmar_ratio(returns, equity)
    assert result > 0


def test_win_rate_counts_positive_periods_only():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.0])
    assert win_rate(returns) == pytest.approx(2 / 4)  # zero excluded from denominator


def test_profit_factor_above_one_when_gains_exceed_losses():
    returns = pd.Series([0.05, -0.01, 0.03, -0.01])
    result = profit_factor(returns)
    assert result == pytest.approx(0.08 / 0.02)


def test_profit_factor_infinite_with_no_losses():
    returns = pd.Series([0.01, 0.02, 0.0])
    assert profit_factor(returns) == float("inf")
