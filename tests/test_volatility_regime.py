import numpy as np
import pandas as pd

from stockanalysis.regime import (
    HIGH,
    LOW,
    MEDIUM,
    classify_volatility_regime,
    realized_volatility,
    regime_summary,
)


def test_realized_volatility_is_non_negative(close):
    from stockanalysis.stats import daily_returns

    vol = realized_volatility(daily_returns(close)).dropna()
    assert (vol >= 0).all()


def test_realized_volatility_higher_in_a_choppier_window():
    calm = pd.Series(np.full(50, 0.001))
    choppy = pd.Series(np.tile([0.03, -0.03], 25))
    calm_vol = realized_volatility(calm, window=20).dropna().iloc[-1]
    choppy_vol = realized_volatility(choppy, window=20).dropna().iloc[-1]
    assert choppy_vol > calm_vol


def test_classify_volatility_regime_returns_all_three_labels():
    rng = np.random.default_rng(0)
    # a return series that clearly alternates between calm and turbulent
    # stretches, so all three regimes should show up
    calm = rng.normal(0, 0.002, 300)
    turbulent = rng.normal(0, 0.04, 300)
    returns = pd.Series(np.concatenate([calm, turbulent, calm]))

    regime, thresholds = classify_volatility_regime(returns, window=21)
    labels = set(regime.dropna().unique())
    assert labels == {LOW, MEDIUM, HIGH}
    assert thresholds.low_pct < thresholds.high_pct


def test_high_regime_days_have_higher_realized_vol_than_low_regime_days():
    rng = np.random.default_rng(1)
    calm = rng.normal(0, 0.002, 300)
    turbulent = rng.normal(0, 0.04, 300)
    returns = pd.Series(np.concatenate([calm, turbulent, calm]))

    regime, _ = classify_volatility_regime(returns, window=21)
    summary = regime_summary(regime, returns)
    assert summary.loc[HIGH, "std_return"] > summary.loc[LOW, "std_return"]


def test_regime_summary_columns():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0, 0.01, 500))
    regime, _ = classify_volatility_regime(returns, window=21)
    summary = regime_summary(regime, returns)
    assert list(summary.columns) == ["count", "mean_return", "std_return"]
    assert list(summary.index) == [LOW, MEDIUM, HIGH]
