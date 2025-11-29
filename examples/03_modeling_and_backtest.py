"""
Forecasting with walk-forward validation, plus a backtest.

    python examples/03_modeling_and_backtest.py
"""

from __future__ import annotations

from stockanalysis.backtest import buy_and_hold, run_backtest, sma_crossover_signal
from stockanalysis.data import SyntheticSource
from stockanalysis.models import (
    BaselinePersistenceModel,
    LinearReturnModel,
    build_lagged_features,
    walk_forward_validate,
)
from stockanalysis.stats import daily_returns
from stockanalysis.viz import plot_equity_curve


def main() -> None:
    ohlcv = SyntheticSource(seed=42).fetch("SYNTH", start="2018-01-01")
    close = ohlcv["Close"]
    returns = daily_returns(close)

    # --- forecasting: does a linear model beat naive persistence? ---
    X, y = build_lagged_features(returns, lags=5)

    baseline_folds = walk_forward_validate(BaselinePersistenceModel, X, y, n_splits=5)
    linear_folds = walk_forward_validate(LinearReturnModel, X, y, n_splits=5)

    baseline_rmse = sum(f.rmse for f in baseline_folds) / len(baseline_folds)
    linear_rmse = sum(f.rmse for f in linear_folds) / len(linear_folds)

    print("Walk-forward validation (5 folds), average RMSE of next-day return:")
    print(f"  Baseline (persistence): {baseline_rmse:.5f}")
    print(f"  Linear model:           {linear_rmse:.5f}")
    verdict = "beats" if linear_rmse < baseline_rmse else "does not beat"
    print(f"  -> linear model {verdict} the naive baseline out of sample")
    print(
        "  (not strong evidence of real predictive skill on synthetic i.i.d. "
        "returns — persistence is a weak baseline here since it compounds "
        "two independent noise terms, while a linear fit shrinks toward the "
        "mean. A fairer bar is a 'predict the historical mean' baseline.)"
    )

    # --- backtesting: SMA crossover vs buy-and-hold ---
    signal = sma_crossover_signal(close, fast=20, slow=50)
    strategy = run_backtest(close, signal, initial_capital=10_000, transaction_cost_bps=5)
    benchmark = buy_and_hold(close, initial_capital=10_000)

    print("\nBacktest: 20/50 SMA crossover vs. buy-and-hold")
    print(f"  Strategy:  {strategy.summary()}")
    print(f"  Benchmark: {benchmark.summary()}")

    fig = plot_equity_curve(strategy.equity_curve, benchmark.equity_curve,
                             title="SMA Crossover vs. Buy & Hold")
    fig.savefig("equity_curve.png", dpi=150)
    print("\nSaved equity_curve.png")


if __name__ == "__main__":
    main()
