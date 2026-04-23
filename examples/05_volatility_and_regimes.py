"""
Volatility deep-dive: stationarity/autocorrelation/normality diagnostics,
rolling volatility regimes, GARCH forecasting, and volatility-targeted
position sizing vs. a flat position of the same signal.

    python examples/05_volatility_and_regimes.py
"""

from __future__ import annotations

from stockanalysis.backtest import (
    run_backtest,
    sma_crossover_signal,
    volatility_target_weights,
)
from stockanalysis.data import SyntheticSource
from stockanalysis.regime import classify_volatility_regime, regime_summary
from stockanalysis.risk.garch import fit_garch, garch_var
from stockanalysis.risk.var import parametric_var
from stockanalysis.stats import daily_returns
from stockanalysis.stats.diagnostics import adf_test, jarque_bera_test, ljung_box_test


def main() -> None:
    ohlcv = SyntheticSource(seed=42).fetch("SYNTH", start="2017-01-01")
    close = ohlcv["Close"]
    returns = daily_returns(close)

    print("Diagnostics on daily returns:")
    for result in (adf_test(returns), ljung_box_test(returns), jarque_bera_test(returns)):
        print(f"  {result.summary()}")

    print("\nVolatility regimes (33rd/67th percentile of 21-day realized vol):")
    regime, thresholds = classify_volatility_regime(returns)
    print(f"  Cutoffs: low <= {thresholds.low_pct:.2%}, high >= {thresholds.high_pct:.2%}")
    print(regime_summary(regime, returns).to_string(float_format="{:.5f}".format))

    print("\nGARCH(1,1) vs. flat-window VaR (95%, 1-day):")
    garch = garch_var(returns, confidence=0.95)
    flat = parametric_var(returns, confidence=0.95)
    print(f"  GARCH VaR:  {garch:.2%}")
    print(f"  Flat VaR:   {flat:.2%}")
    forecast = fit_garch(returns, horizon_days=5).annualized_forecast()
    print(f"  5-day annualized vol forecast path: {[f'{v:.1%}' for v in forecast]}")

    print("\nSMA crossover: flat position vs. volatility-targeted position:")
    signal = sma_crossover_signal(close, fast=20, slow=50)
    flat_result = run_backtest(close, signal, transaction_cost_bps=5)
    weights = volatility_target_weights(signal, returns, target_annual_vol=0.15)
    targeted_result = run_backtest(close, weights, transaction_cost_bps=5)
    print(f"  Flat position:        {flat_result.summary()}")
    print(f"  Vol-targeted position: {targeted_result.summary()}")


if __name__ == "__main__":
    main()
