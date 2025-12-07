"""Volume and range-based indicators. Need OHLCV, not just Close."""

from __future__ import annotations

import numpy as np
import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: running total of volume, signed by the day's
    price direction. Flat days add nothing."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum().rename("OBV")


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Cumulative Volume-Weighted Average Price (from the start of the series,
    not a rolling window - reset per session if you want the intraday version)."""
    typical_price = (high + low + close) / 3
    cum_vol = volume.cumsum()
    cum_pv = (typical_price * volume).cumsum()
    return (cum_pv / cum_vol).rename("VWAP")


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1).rename("TR")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing) - volatility measured in price
    units rather than percent, useful for stop distances and position sizing."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean().rename(
        f"ATR_{window}"
    )


def stochastic_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_window: int = 14, d_window: int = 3,
) -> pd.DataFrame:
    """%K and %D. %K = where close sits in the recent high/low range (0-100),
    %D is a 3-day SMA of %K used as the signal line."""
    lowest_low = low.rolling(window=k_window, min_periods=k_window).min()
    highest_high = high.rolling(window=k_window, min_periods=k_window).max()
    percent_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    percent_d = percent_k.rolling(window=d_window, min_periods=d_window).mean()
    return pd.DataFrame({"%K": percent_k, "%D": percent_d})
