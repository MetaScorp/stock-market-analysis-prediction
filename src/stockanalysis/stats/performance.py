"""
Performance metrics beyond plain Sharpe. backtest.engine already reports
total/annualized return and max drawdown - this fills in the metrics that
matter once you're comparing strategies against each other, not just
against buy-and-hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Drawdown at each point: how far below the running peak, as a fraction."""
    running_max = equity_curve.cummax()
    return (equity_curve / running_max - 1).rename("drawdown")


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """Longest stretch (in bars) spent below a prior peak before making a
    new high. Not the same as the size of the drawdown - a small drawdown
    that never recovers can last longer than a big one that snaps back."""
    running_max = equity_curve.cummax()
    underwater = equity_curve < running_max

    longest = 0
    current = 0
    for is_under in underwater:
        if is_under:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Like Sharpe, but only penalizes downside deviation - upside volatility
    isn't "risk" in the way Sharpe's plain std dev treats it."""
    period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - period_rf
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    if downside_std == 0 or np.isnan(downside_std):
        return float("nan")
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def calmar_ratio(
    returns: pd.Series, equity_curve: pd.Series, periods_per_year: int = 252
) -> float:
    """Annualized return divided by max drawdown (as a positive number).
    Penalizes strategies that make their return by taking one huge hit."""
    from stockanalysis.stats.returns import annualized_return

    ann_return = annualized_return(returns, periods_per_year)
    mdd = abs(drawdown_series(equity_curve).min())
    if mdd == 0:
        return float("nan")
    return float(ann_return / mdd)


def win_rate(returns: pd.Series) -> float:
    """Fraction of non-zero-return periods that were positive."""
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return float("nan")
    return float((nonzero > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Gross gains divided by gross losses. >1 means winners outweigh losers
    in dollar terms, independent of how often you win."""
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float("nan") if gains == 0 else float("inf")
    return float(gains / losses)
