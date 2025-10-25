import numpy as np
import pandas as pd
import pytest

from stockanalysis.indicators import bollinger_bands, ema, macd, rsi, sma


def test_sma_known_values():
    prices = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = sma(prices, window=3)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(3.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_ema_reacts_faster_than_long_sma_to_a_shock(close):
    shocked = close.copy()
    shocked.iloc[-1] *= 1.5
    fast_ema = ema(shocked, span=5)
    slow_sma = sma(shocked, window=50)
    ema_move = abs(fast_ema.iloc[-1] - fast_ema.iloc[-2])
    sma_move = abs(slow_sma.iloc[-1] - slow_sma.iloc[-2])
    assert ema_move > sma_move


def test_rsi_is_bounded(close):
    result = rsi(close, window=14).dropna()
    assert (result >= 0).all()
    assert (result <= 100).all()


def test_rsi_is_100_for_strictly_increasing_series():
    prices = pd.Series(np.arange(1, 30, dtype=float))
    result = rsi(prices, window=14).dropna()
    assert (result == 100.0).all()


def test_macd_returns_expected_columns(close):
    result = macd(close)
    assert list(result.columns) == ["macd", "signal", "histogram"]
    valid = result.dropna()
    assert np.allclose(valid["histogram"], valid["macd"] - valid["signal"])


def test_bollinger_bands_ordering(close):
    bands = bollinger_bands(close, window=20).dropna()
    assert (bands["upper"] >= bands["middle"]).all()
    assert (bands["middle"] >= bands["lower"]).all()
