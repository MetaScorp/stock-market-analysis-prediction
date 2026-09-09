"""
Real data case study: runs the pipeline against actual historical AAPL
data (1987, including Black Monday) instead of synthetic data.

Every other example in this folder defaults to SyntheticSource, which is
i.i.d. by construction - useful for testing, but it will never show fat
tails, autocorrelation, or a real crash. This script exists to check that
against a real series, on record, without needing network access.

    python examples/07_real_data_case_study.py
"""

from __future__ import annotations

from pathlib import Path

from stockanalysis.backtest import buy_and_hold, run_backtest, sma_crossover_signal
from stockanalysis.data import CSVSource, check_data_quality
from stockanalysis.regime import fit_markov_switching
from stockanalysis.risk.garch import garch_var
from stockanalysis.risk.var import historical_var, parametric_var
from stockanalysis.stats import (
    annualized_return,
    annualized_volatility,
    daily_returns,
    sharpe_ratio,
)
from stockanalysis.stats.diagnostics import adf_test, jarque_bera_test, ljung_box_test

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    ohlcv = CSVSource(DATA_DIR).fetch("AAPL_1987", start="1987-01-01", end="1987-12-31")
    close = ohlcv["Close"]
    returns = daily_returns(close)

    print(f"AAPL, {close.index[0].date()} to {close.index[-1].date()} ({len(ohlcv)} real trading days)")
    print("Real market data, not synthetic - results below will look different from")
    print("the other examples, and that's the point.\n")

    quality = check_data_quality(ohlcv)
    print(f"Data quality: {quality.summary()}")
    print(
        "(the missing days are real market holidays, not data errors - the "
        "checker doesn't know a holiday calendar, it just flags gaps)\n"
    )

    print(f"Annualized return:     {annualized_return(returns):.2%}")
    print(f"Annualized volatility: {annualized_volatility(returns):.2%}")
    print(f"Sharpe ratio:          {sharpe_ratio(returns):.2f}\n")

    print("Diagnostics - contrast this with the synthetic examples, where none")
    print("of these three normally reject:")
    for result in (adf_test(returns), ljung_box_test(returns), jarque_bera_test(returns)):
        print(f"  {result.summary()}")

    print("\nVaR (95%, 1-day):")
    print(f"  Historical:  {historical_var(returns):.2%}")
    print(f"  Parametric:  {parametric_var(returns):.2%}")
    try:
        print(f"  GARCH:       {garch_var(returns):.2%}")
    except Exception as exc:  # noqa: BLE001 - GARCH fit can fail to converge
        print(f"  GARCH:       fit failed ({exc})")

    print("\nRegime detection (Markov-switching):")
    markov = fit_markov_switching(returns)
    print(f"  {markov.summary()}".replace("\n", "\n  "))

    print("\nBacktest: SMA(10/30) crossover vs. buy-and-hold")
    signal = sma_crossover_signal(close, fast=10, slow=30)
    strategy = run_backtest(close, signal, transaction_cost_bps=20)
    benchmark = buy_and_hold(close)
    print(f"  Strategy:  {strategy.summary()}")
    print(f"  Benchmark: {benchmark.summary()}")
    print(
        "\n  Buy-and-hold wins this particular year by a wide margin - 1987 was a "
        "strong year for the stock overall, crash included, and a slow-moving "
        "SMA crossover mostly just adds cost and lag on top of that. That's a "
        "real result on one ticker in one year, not a general claim about "
        "either approach."
    )


if __name__ == "__main__":
    main()
