"""
Is a backtest result actually different from noise?

A single Sharpe ratio number from one historical path tells you nothing
about how much of it is luck. Two ways to get at that here:

- bootstrap_sharpe_ci: resample the return series (with replacement) many
  times and refit the Sharpe ratio each time, to get a confidence interval
  instead of a point estimate.
- permutation_test: shuffle which days the strategy was in the market
  (breaking the relationship between signal and subsequent return while
  keeping the same set of position days) and see how often a random
  version of the same signal would have done as well. This is the
  strategy-evaluation version of a permutation test - it asks "is this
  edge distinguishable from randomly picking which days to be long?", not
  "is this a good absolute return."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockanalysis.stats.returns import sharpe_ratio


@dataclass
class BootstrapResult:
    point_estimate: float
    lower: float
    upper: float
    confidence: float
    samples: np.ndarray

    def summary(self) -> str:
        return (
            f"Sharpe: {self.point_estimate:.2f} "
            f"[{self.confidence:.0%} CI: {self.lower:.2f}, {self.upper:.2f}]"
        )


def bootstrap_sharpe_ci(
    returns: pd.Series,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    periods_per_year: int = 252,
    seed: int = 42,
) -> BootstrapResult:
    """Block-free i.i.d. bootstrap of the Sharpe ratio. Resampling single
    days with replacement understates uncertainty if returns are
    autocorrelated (check stats.diagnostics.ljung_box_test first) - for
    strongly autocorrelated series a block bootstrap would be more honest,
    but that's not implemented here."""
    rng = np.random.default_rng(seed)
    values = returns.dropna().to_numpy()
    n = len(values)

    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resampled = rng.choice(values, size=n, replace=True)
        samples[i] = sharpe_ratio(pd.Series(resampled), periods_per_year=periods_per_year)

    alpha = 1 - confidence
    lower, upper = np.nanquantile(samples, [alpha / 2, 1 - alpha / 2])

    return BootstrapResult(
        point_estimate=sharpe_ratio(returns, periods_per_year=periods_per_year),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        samples=samples,
    )


@dataclass
class PermutationResult:
    observed_sharpe: float
    null_sharpes: np.ndarray
    p_value: float

    def summary(self) -> str:
        return (
            f"Observed Sharpe: {self.observed_sharpe:.2f} | "
            f"null mean: {np.nanmean(self.null_sharpes):.2f} | "
            f"p-value: {self.p_value:.4f} "
            f"({'not ' if self.p_value >= 0.05 else ''}distinguishable from a "
            f"random signal with the same number of active days at 5%)"
        )


def permutation_test_signal(
    signal: pd.Series,
    market_returns: pd.Series,
    n_permutations: int = 2000,
    periods_per_year: int = 252,
    seed: int = 42,
) -> PermutationResult:
    """Shuffle *when* the signal is active (same number of long days, random
    order), re-run the backtest math, and compare the observed Sharpe
    against that null distribution. A low p-value means it's unlikely a
    random selection of the same number of days would have done this well
    - it does not mean the strategy will keep working."""
    from stockanalysis.backtest.engine import run_backtest

    rng = np.random.default_rng(seed)
    observed = run_backtest(
        market_returns.add(1).cumprod(), signal, periods_per_year=periods_per_year
    ).sharpe_ratio

    signal_values = signal.to_numpy()
    null_sharpes = np.empty(n_permutations)
    prices = market_returns.add(1).cumprod()
    for i in range(n_permutations):
        shuffled = rng.permutation(signal_values)
        shuffled_signal = pd.Series(shuffled, index=signal.index)
        result = run_backtest(prices, shuffled_signal, periods_per_year=periods_per_year)
        null_sharpes[i] = result.sharpe_ratio

    valid_null = null_sharpes[~np.isnan(null_sharpes)]
    p_value = float((valid_null >= observed).mean()) if len(valid_null) else float("nan")

    return PermutationResult(
        observed_sharpe=observed, null_sharpes=null_sharpes, p_value=p_value
    )
