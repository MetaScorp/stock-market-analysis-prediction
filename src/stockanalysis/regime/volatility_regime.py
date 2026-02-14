"""
Volatility regime classification.

Nothing fancy here: rolling realized vol, compared against its own
historical percentiles, split into low/medium/high buckets. It's a
reasonable first cut before reaching for something like a Markov
regime-switching model (see regime.markov_switching) - cheap to compute,
easy to explain, and a decent baseline to check a fancier model against.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

LOW, MEDIUM, HIGH = "low", "medium", "high"


def realized_volatility(
    returns: pd.Series, window: int = 21, periods_per_year: int = 252
) -> pd.Series:
    """Annualized rolling realized volatility."""
    return (returns.rolling(window=window, min_periods=window).std()
            * (periods_per_year ** 0.5)).rename("realized_vol")


@dataclass
class RegimeThresholds:
    low_pct: float
    high_pct: float


def classify_volatility_regime(
    returns: pd.Series,
    window: int = 21,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
    periods_per_year: int = 252,
) -> tuple[pd.Series, RegimeThresholds]:
    """Label each day low/medium/high vol based on where rolling realized
    vol sits relative to its own history (quantiles computed in-sample -
    this is descriptive, not something to trade on directly without
    checking it holds out of sample too)."""
    vol = realized_volatility(returns, window=window, periods_per_year=periods_per_year)
    valid = vol.dropna()
    low_cut = float(valid.quantile(low_quantile))
    high_cut = float(valid.quantile(high_quantile))

    def _label(x: float) -> str:
        if pd.isna(x):
            return pd.NA
        if x <= low_cut:
            return LOW
        if x >= high_cut:
            return HIGH
        return MEDIUM

    regime = vol.apply(_label).rename("vol_regime")
    return regime, RegimeThresholds(low_pct=low_cut, high_pct=high_cut)


def regime_summary(regime: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """Mean/vol/count of returns conditioned on regime - a quick way to
    see whether the buckets actually mean anything for this series."""
    aligned = pd.DataFrame({"regime": regime, "return": returns}).dropna()
    grouped = aligned.groupby("regime")["return"]
    return pd.DataFrame(
        {
            "count": grouped.count(),
            "mean_return": grouped.mean(),
            "std_return": grouped.std(),
        }
    ).reindex([LOW, MEDIUM, HIGH])
