import numpy as np
import pandas as pd

from stockanalysis.backtest import run_backtest, volatility_target_weights


def _dates(n):
    return pd.bdate_range("2022-01-03", periods=n)


def test_weights_are_zero_where_signal_is_zero():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 100), index=_dates(100))
    signal = pd.Series(0, index=_dates(100))
    weights = volatility_target_weights(signal, returns, window=20)
    assert (weights == 0).all()


def test_weights_are_capped_at_max_leverage():
    # very low realized vol -> scale would blow past the cap without clipping
    returns = pd.Series(np.full(100, 0.0001), index=_dates(100))
    signal = pd.Series(1, index=_dates(100))
    weights = volatility_target_weights(
        signal, returns, target_annual_vol=0.5, window=20, max_leverage=2.0
    )
    assert weights.dropna().max() <= 2.0


def test_higher_volatility_leads_to_smaller_position():
    rng = np.random.default_rng(1)
    calm_returns = pd.Series(rng.normal(0, 0.005, 100), index=_dates(100))
    choppy_returns = pd.Series(rng.normal(0, 0.03, 100), index=_dates(100))
    signal = pd.Series(1, index=_dates(100))

    calm_weight = volatility_target_weights(signal, calm_returns, window=20).iloc[-1]
    choppy_weight = volatility_target_weights(signal, choppy_returns, window=20).iloc[-1]
    assert calm_weight > choppy_weight


def test_vol_targeted_backtest_produces_lower_realized_vol_than_binary_signal():
    rng = np.random.default_rng(2)
    prices = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, 500))), index=_dates(500)
    )
    market_returns = prices.pct_change().dropna()
    always_on = pd.Series(1, index=prices.index)

    plain = run_backtest(prices, always_on)
    weights = volatility_target_weights(always_on, market_returns, window=20)
    targeted = run_backtest(prices, weights)

    assert targeted.annualized_volatility < plain.annualized_volatility
