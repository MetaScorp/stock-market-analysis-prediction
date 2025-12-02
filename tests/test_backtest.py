import pandas as pd
import pytest

from stockanalysis.backtest import buy_and_hold, run_backtest, sma_crossover_signal


def test_run_backtest_flat_signal_yields_flat_equity():
    prices = pd.Series([100, 110, 90, 120], dtype=float,
                        index=pd.bdate_range("2022-01-03", periods=4))
    flat_signal = pd.Series(0, index=prices.index)
    result = run_backtest(prices, flat_signal, initial_capital=1000.0)
    assert (result.equity_curve == 1000.0).all()
    assert result.total_return == pytest.approx(0.0)


def test_buy_and_hold_matches_raw_price_growth():
    prices = pd.Series([100, 110, 121], dtype=float,
                        index=pd.bdate_range("2022-01-03", periods=3))
    result = buy_and_hold(prices, initial_capital=1000.0)
    expected_total_return = 121 / 100 - 1
    assert result.total_return == pytest.approx(expected_total_return)


def test_signal_is_lagged_one_day_to_avoid_lookahead():
    prices = pd.Series([100, 100, 200], dtype=float,
                        index=pd.bdate_range("2022-01-03", periods=3))
    signal = pd.Series([0, 0, 1], index=prices.index)
    result = run_backtest(prices, signal, initial_capital=1000.0)
    assert result.total_return == pytest.approx(0.0)


def test_transaction_costs_reduce_returns():
    prices = pd.Series([100, 110, 100, 110, 100], dtype=float,
                        index=pd.bdate_range("2022-01-03", periods=5))
    signal = pd.Series([1, 0, 1, 0, 1], index=prices.index)
    free = run_backtest(prices, signal, initial_capital=1000.0, transaction_cost_bps=0)
    costly = run_backtest(prices, signal, initial_capital=1000.0, transaction_cost_bps=50)
    assert costly.total_return < free.total_return


def test_max_drawdown_is_non_positive(close):
    result = buy_and_hold(close)
    assert result.max_drawdown <= 0


def test_sma_crossover_signal_is_binary(close):
    signal = sma_crossover_signal(close, fast=10, slow=30)
    assert set(signal.dropna().unique()).issubset({0, 1})
