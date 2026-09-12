"""
Everything else in this suite runs on SyntheticSource, which is i.i.d. by
construction. That's good for testing individual functions, but it can't
catch anything that only shows up on real market data - fat tails, real
autocorrelation, actual data gaps. This runs the pipeline against the
bundled real AAPL 1987 data (see examples/data/README.md) as a sanity
check that it holds up on something that wasn't generated to be well-
behaved.
"""

import math
from pathlib import Path

import pandas as pd

from stockanalysis.data import CSVSource, check_data_quality
from stockanalysis.regime import fit_markov_switching
from stockanalysis.risk.var import historical_var, parametric_var
from stockanalysis.stats import daily_returns, sharpe_ratio
from stockanalysis.stats.diagnostics import adf_test, jarque_bera_test, ljung_box_test

DATA_DIR = Path(__file__).parent.parent / "examples" / "data"


def _load():
    return CSVSource(DATA_DIR).fetch("AAPL_1987", start="1987-01-01", end="1987-12-31")


def test_real_fixture_loads_with_expected_shape():
    ohlcv = _load()
    assert len(ohlcv) == 253
    assert list(ohlcv.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_real_fixture_flags_black_monday_as_an_extreme_move():
    ohlcv = _load()
    report = check_data_quality(ohlcv)
    assert not report.is_clean
    assert pd.Timestamp("1987-10-19") in report.extreme_move_dates


def test_real_fixture_flags_market_holidays_as_missing_days():
    ohlcv = _load()
    report = check_data_quality(ohlcv)
    # not a data error - genuine market closures (New Year's, Good Friday,
    # Memorial Day, July 4th, Labor Day, Thanksgiving, Christmas)
    assert len(report.missing_business_days) == 7


def test_real_returns_reject_normality_unlike_synthetic_data():
    returns = daily_returns(_load()["Close"])
    result = jarque_bera_test(returns)
    assert result.reject_null_at_5pct is True


def test_real_returns_show_significant_autocorrelation():
    # unlike the i.i.d. synthetic generator - a real, if unremarkable, way
    # real markets differ from the toy data used everywhere else in this suite
    returns = daily_returns(_load()["Close"])
    result = ljung_box_test(returns)
    assert result.reject_null_at_5pct is True


def test_prices_are_non_stationary_but_returns_are():
    ohlcv = _load()
    close = ohlcv["Close"]
    returns = daily_returns(close)
    assert adf_test(close).reject_null_at_5pct is False
    assert adf_test(returns).reject_null_at_5pct is True


def test_parametric_var_and_historical_var_are_both_positive_and_finite():
    returns = daily_returns(_load()["Close"])
    hist = historical_var(returns)
    param = parametric_var(returns)
    assert 0 < hist < 1
    assert 0 < param < 1


def test_markov_switching_converges_and_finds_a_high_vol_regime():
    returns = daily_returns(_load()["Close"])
    result = fit_markov_switching(returns)
    assert result.converged is True
    assert max(result.regime_std) > 0.05  # the crash period should dominate one regime


def test_sharpe_ratio_is_a_finite_number_on_real_data():
    returns = daily_returns(_load()["Close"])
    result = sharpe_ratio(returns)
    assert not math.isnan(result)
    assert abs(result) < 100  # sanity bound, not a strict claim
