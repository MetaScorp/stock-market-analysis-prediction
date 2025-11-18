"""
Backtesting.

Turns a signal into positions, applies a one-day execution lag so you can't
trade on the same day you observe the signal, and optionally charges
transaction costs. Always report buy-and-hold alongside a strategy's
result — a return number means nothing without a benchmark next to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockanalysis.indicators.technical import sma
from stockanalysis.stats.returns import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float

    def summary(self) -> str:
        return (
            f"Total return: {self.total_return:.2%} | "
            f"Annualized return: {self.annualized_return:.2%} | "
            f"Annualized vol: {self.annualized_volatility:.2%} | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"Max drawdown: {self.max_drawdown:.2%}"
        )


def sma_crossover_signal(
    prices: pd.Series, fast: int = 20, slow: int = 50
) -> pd.Series:
    """+1 while fast SMA > slow SMA, else 0 (long-only, flat otherwise)."""
    fast_sma = sma(prices, window=fast)
    slow_sma = sma(prices, window=slow)
    signal = (fast_sma > slow_sma).astype(int)
    return signal.rename("signal")


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def run_backtest(
    prices: pd.Series,
    signal: pd.Series,
    initial_capital: float = 10_000.0,
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Long/flat backtest. signal is shifted one day before being applied,
    so a signal computed off day t's close only acts starting day t+1."""
    market_returns = prices.pct_change().fillna(0)
    position = signal.shift(1).fillna(0).reindex(market_returns.index).fillna(0)

    strategy_returns = position * market_returns

    turnover = position.diff().abs().fillna(0)
    cost = turnover * (transaction_cost_bps / 10_000)
    strategy_returns = strategy_returns - cost

    equity_curve = initial_capital * (1 + strategy_returns).cumprod()
    equity_curve.name = "equity"

    total_return = float(equity_curve.iloc[-1] / initial_capital - 1)

    return BacktestResult(
        equity_curve=equity_curve,
        returns=strategy_returns,
        total_return=total_return,
        annualized_return=annualized_return(strategy_returns, periods_per_year),
        annualized_volatility=annualized_volatility(strategy_returns, periods_per_year),
        sharpe_ratio=sharpe_ratio(strategy_returns, periods_per_year=periods_per_year),
        max_drawdown=_max_drawdown(equity_curve),
    )


def buy_and_hold(
    prices: pd.Series, initial_capital: float = 10_000.0, periods_per_year: int = 252
) -> BacktestResult:
    """The mandatory benchmark — what if you'd just held the whole time."""
    always_long = pd.Series(1, index=prices.index)
    return run_backtest(
        prices, always_long, initial_capital=initial_capital,
        periods_per_year=periods_per_year,
    )
