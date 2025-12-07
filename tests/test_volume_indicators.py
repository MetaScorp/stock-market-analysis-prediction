import pandas as pd

from stockanalysis.indicators import atr, obv, stochastic_oscillator, true_range, vwap


def test_obv_accumulates_signed_volume():
    close = pd.Series([10.0, 11.0, 10.5, 10.5, 12.0])
    volume = pd.Series([100, 200, 150, 50, 300])
    result = obv(close, volume)
    # day0: no prior diff -> 0, day1: up -> +200, day2: down -> -150,
    # day3: flat -> +0, day4: up -> +300
    expected = pd.Series([0.0, 200.0, 50.0, 50.0, 350.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_vwap_between_high_and_low(ohlcv):
    result = vwap(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], ohlcv["Volume"])
    assert (result >= ohlcv["Low"].min()).all()
    assert (result <= ohlcv["High"].max()).all()


def test_true_range_is_at_least_the_days_range(ohlcv):
    tr = true_range(ohlcv["High"], ohlcv["Low"], ohlcv["Close"])
    day_range = ohlcv["High"] - ohlcv["Low"]
    assert (tr >= day_range - 1e-9).all()


def test_atr_is_non_negative(ohlcv):
    result = atr(ohlcv["High"], ohlcv["Low"], ohlcv["Close"]).dropna()
    assert (result >= 0).all()


def test_stochastic_oscillator_is_bounded(ohlcv):
    result = stochastic_oscillator(ohlcv["High"], ohlcv["Low"], ohlcv["Close"]).dropna()
    assert (result["%K"] >= 0).all() and (result["%K"] <= 100).all()
    assert (result["%D"] >= 0).all() and (result["%D"] <= 100).all()
