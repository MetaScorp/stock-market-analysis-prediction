"""
Cross-ticker analysis: rolling correlation and beta against a benchmark.

Static correlation (stats.correlation_matrix) tells you the relationship
averaged over the whole window. It hides regime changes - two stocks can
be correlated 0.8 in a calm market and 0.2 in a crash. Rolling versions
show that.
"""

from __future__ import annotations

import pandas as pd


def rolling_correlation(
    returns_a: pd.Series, returns_b: pd.Series, window: int = 60
) -> pd.Series:
    aligned_a, aligned_b = returns_a.align(returns_b, join="inner")
    return aligned_a.rolling(window=window, min_periods=window).corr(
        aligned_b
    ).rename("rolling_corr")


def rolling_beta(
    asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 60
) -> pd.Series:
    """Rolling OLS beta of asset vs. benchmark: cov(asset, bench) / var(bench)."""
    asset, bench = asset_returns.align(benchmark_returns, join="inner")
    cov = asset.rolling(window=window, min_periods=window).cov(bench)
    var = bench.rolling(window=window, min_periods=window).var()
    return (cov / var).rename("rolling_beta")


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Single beta value over the full overlapping period."""
    asset, bench = asset_returns.align(benchmark_returns, join="inner")
    covariance = asset.cov(bench)
    variance = bench.var()
    if variance == 0:
        return float("nan")
    return float(covariance / variance)


def relative_strength(
    prices: pd.Series, benchmark_prices: pd.Series
) -> pd.Series:
    """Price ratio vs. a benchmark, rebased to 1.0 at the start of the
    overlap. Rising = outperforming the benchmark, regardless of whether
    both are up or down in absolute terms."""
    asset, bench = prices.align(benchmark_prices, join="inner")
    ratio = asset / bench
    return (ratio / ratio.iloc[0]).rename("relative_strength")
