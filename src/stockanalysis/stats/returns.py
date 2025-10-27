"""Return series and the usual risk-adjusted performance stats."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(prices: pd.Series, log: bool = False) -> pd.Series:
    """Daily simple or log returns. First value is dropped."""
    if log:
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    return returns.dropna().rename("return")


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Growth of $1 given a return series."""
    return (1 + returns).cumprod().rename("cumulative_return")


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Geometric annualized return."""
    total_growth = (1 + returns).prod()
    n_periods = len(returns)
    if n_periods == 0:
        return float("nan")
    return float(total_growth ** (periods_per_year / n_periods) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Annualized Sharpe ratio. risk_free_rate is annualized, converted
    to a per-period rate before subtracting it from returns."""
    period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - period_rf
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Pairwise return correlation across columns of a price DataFrame."""
    returns = prices.pct_change().dropna(how="all")
    return returns.corr()
