import numpy as np
import pandas as pd

from stockanalysis.stats.diagnostics import adf_test, jarque_bera_test, ljung_box_test


def test_adf_does_not_reject_stationarity_for_a_random_walk():
    rng = np.random.default_rng(0)
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)))
    result = adf_test(prices)
    assert result.reject_null_at_5pct is False


def test_adf_rejects_non_stationarity_for_white_noise():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 500))
    result = adf_test(returns)
    assert result.reject_null_at_5pct is True


def test_ljung_box_does_not_reject_for_iid_noise():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0, 0.01, 1000))
    result = ljung_box_test(returns, lags=10)
    assert result.reject_null_at_5pct is False


def test_ljung_box_rejects_for_strongly_autocorrelated_series():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, 1000)
    autocorrelated = pd.Series(noise).rolling(5).mean().dropna()
    result = ljung_box_test(autocorrelated, lags=10)
    assert result.reject_null_at_5pct is True


def test_jarque_bera_does_not_reject_for_normal_data():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0, 0.01, 2000))
    result = jarque_bera_test(returns)
    assert result.reject_null_at_5pct is False


def test_jarque_bera_rejects_for_fat_tailed_data():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.standard_t(df=2, size=2000) * 0.01)
    result = jarque_bera_test(returns)
    assert result.reject_null_at_5pct is True


def test_summary_mentions_verdict():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0, 0.01, 300))
    text = adf_test(returns).summary()
    assert "H0" in text and "%" in text
