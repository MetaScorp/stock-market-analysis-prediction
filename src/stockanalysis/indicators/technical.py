"""Technical indicators. All vectorized, work on plain Series/DataFrame."""

from __future__ import annotations

import pandas as pd


def sma(prices: pd.Series, window: int = 20) -> pd.Series:
    """Simple moving average."""
    return prices.rolling(window=window, min_periods=window).mean().rename(
        f"SMA_{window}"
    )


def ema(prices: pd.Series, span: int = 20) -> pd.Series:
    """Exponentially weighted moving average."""
    return prices.ewm(span=span, adjust=False).mean().rename(f"EMA_{span}")


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index, Wilder's smoothing via EWM."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    result = 100 - (100 / (1 + rs))
    result = result.where(avg_loss != 0, 100.0)  # no down days -> RSI 100
    return result.rename(f"RSI_{window}")


def macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD. Returns a DataFrame with macd/signal/histogram columns."""
    fast_ema = ema(prices, span=fast)
    slow_ema = ema(prices, span=slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def bollinger_bands(
    prices: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands: middle (SMA), upper, lower."""
    middle = sma(prices, window=window)
    std = prices.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower})
