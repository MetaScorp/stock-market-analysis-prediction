"""
Research report: runs the full pipeline on one ticker and prints a summary
with explicit caveats. This is meant to be read critically, not treated as
a signal - the point is that every number here can be recomputed and
challenged, not that any of it is a recommendation.

    python examples/06_research_report.py
    python examples/06_research_report.py --live AAPL
"""

from __future__ import annotations

import argparse

from stockanalysis.backtest import (
    run_backtest,
    sma_crossover_signal,
    volatility_target_weights,
)
from stockanalysis.data import CachedYFinanceSource, SyntheticSource, check_data_quality
from stockanalysis.evaluation import (
    bootstrap_sharpe_ci,
    compare_models,
    permutation_test_signal,
)
from stockanalysis.models import (
    BaselinePersistenceModel,
    LinearReturnModel,
    build_lagged_features,
    walk_forward_validate,
)
from stockanalysis.regime import (
    classify_volatility_regime,
    fit_markov_switching,
    regime_summary,
)
from stockanalysis.risk.garch import garch_var
from stockanalysis.risk.var import historical_var, parametric_var
from stockanalysis.stats import (
    annualized_return,
    annualized_volatility,
    daily_returns,
    sharpe_ratio,
)
from stockanalysis.stats.diagnostics import adf_test, jarque_bera_test, ljung_box_test


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", metavar="TICKER", default=None)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--permutations", type=int, default=500)
    args = parser.parse_args()

    if args.live:
        ticker = args.live
        ohlcv = CachedYFinanceSource(args.cache_dir).fetch(ticker, start=args.start)
    else:
        ticker = "SYNTH"
        ohlcv = SyntheticSource(seed=7).fetch(ticker, start=args.start)

    close = ohlcv["Close"]
    returns = daily_returns(close)
    n = len(returns)

    print(f"Research report: {ticker}")
    print(f"Range: {close.index[0].date()} to {close.index[-1].date()} ({n} trading days)")
    print(
        "This report describes the historical record and out-of-sample "
        "backtest behavior of one strategy on one series. It is not a "
        "forecast of future returns and none of the numbers below should be "
        "read as investment advice."
    )

    quality = check_data_quality(ohlcv)
    print(f"\nData quality: {quality.summary()}")
    if not quality.is_clean:
        print("Proceeding anyway, but treat the numbers below with that in mind.")

    section("1. Return statistics")
    print(f"Annualized return:     {annualized_return(returns):.2%}")
    print(f"Annualized volatility: {annualized_volatility(returns):.2%}")
    print(f"Sharpe ratio (point):  {sharpe_ratio(returns):.2f}")

    section("2. Distributional diagnostics")
    print("Testing the assumptions behind the stats above, not just reporting them:")
    for result in (adf_test(returns), ljung_box_test(returns), jarque_bera_test(returns)):
        print(f"  {result.summary()}")
    print(
        "If Jarque-Bera rejects normality (it usually does for real return "
        "series), parametric VaR below is understating tail risk relative "
        "to the historical estimate."
    )

    section("3. Value at Risk (95%, 1-day)")
    hist = historical_var(returns)
    param = parametric_var(returns)
    print(f"  Historical VaR:  {hist:.2%}")
    print(f"  Parametric VaR:  {param:.2%}")
    try:
        garch = garch_var(returns)
        print(f"  GARCH VaR:       {garch:.2%}  (reacts to recent volatility, not just the full sample)")
    except Exception as exc:  # noqa: BLE001 - GARCH fit can fail to converge
        print(f"  GARCH VaR:       fit failed ({exc})")

    section("4. Volatility regimes")
    regime, thresholds = classify_volatility_regime(returns)
    print(f"Rolling-quantile regimes (low <= {thresholds.low_pct:.2%}, high >= {thresholds.high_pct:.2%} 21-day realized vol):")
    print(regime_summary(regime, returns).to_string(float_format="{:.5f}".format))

    print("\nMarkov-switching model (estimated regimes and persistence):")
    markov = fit_markov_switching(returns)
    if markov.converged:
        print(markov.summary())
    else:
        print("  did not converge on this series - regimes below are not reliable")

    section("5. Backtest: SMA(20/50) crossover")
    signal = sma_crossover_signal(close, fast=20, slow=50)
    flat = run_backtest(close, signal, transaction_cost_bps=5)
    weights = volatility_target_weights(signal, returns, target_annual_vol=0.15)
    targeted = run_backtest(close, weights, transaction_cost_bps=5)
    print(f"  Flat position:         {flat.summary()}")
    print(f"  Vol-targeted position: {targeted.summary()}")

    section("6. Is the backtest edge distinguishable from noise?")
    boot = bootstrap_sharpe_ci(flat.returns)
    print(f"  Bootstrap {boot.summary()}")
    perm = permutation_test_signal(signal, returns, n_permutations=args.permutations)
    print(f"  Permutation test: {perm.summary()}")
    print(
        "  A wide bootstrap CI or a high permutation p-value means this "
        "backtest result is not strong evidence of a repeatable edge, "
        "regardless of how good the headline return looks."
    )

    section("7. Does a linear model beat naive persistence, out of sample?")
    X, y = build_lagged_features(returns, lags=5)
    baseline_folds = walk_forward_validate(BaselinePersistenceModel, X, y, n_splits=5)
    linear_folds = walk_forward_validate(LinearReturnModel, X, y, n_splits=5)
    comparison = compare_models(baseline_folds, linear_folds)
    print(f"  {comparison.summary()}")
    print(
        "  This compares squared forecast errors on the same walk-forward "
        "test points, paired, not just two RMSE numbers side by side - the "
        "CI is what tells you whether the gap is real or could be luck."
    )

    section("Assumptions and limitations")
    print(
        "- One ticker, one signal, one cost assumption (5 bps/trade). None\n"
        "  of this generalizes to other assets or parameter choices without\n"
        "  re-running it.\n"
        "- The backtest is in-sample: the SMA windows and vol target were\n"
        "  not tuned out-of-sample here, so treat the specific numbers as\n"
        "  illustrative, not optimized.\n"
        "- GARCH and Markov-switching models assume the return-generating\n"
        "  process is roughly stable; both can be misleading after a\n"
        "  genuine structural break.\n"
        "- Statistical significance (sections 6-7) is not economic\n"
        "  significance - a low p-value or a CI that excludes zero doesn't\n"
        "  imply the strategy or model is profitable after realistic costs\n"
        "  and slippage."
    )


if __name__ == "__main__":
    main()
