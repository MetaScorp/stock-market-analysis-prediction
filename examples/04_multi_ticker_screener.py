"""
Multi-ticker screener: pulls a handful of tickers, computes indicators and
risk stats for each, and ranks them. Uses SyntheticSource by default so it
runs offline; pass --live to use cached real data via yfinance.

    python examples/04_multi_ticker_screener.py
    python examples/04_multi_ticker_screener.py --live AAPL MSFT GOOGL AMZN
"""

from __future__ import annotations

import argparse

import pandas as pd

from stockanalysis.data import CachedYFinanceSource, SyntheticSource
from stockanalysis.indicators import atr, rsi
from stockanalysis.stats import (
    annualized_return,
    annualized_volatility,
    daily_returns,
    sharpe_ratio,
)


def analyze_ticker(ticker: str, ohlcv: pd.DataFrame) -> dict:
    close = ohlcv["Close"]
    returns = daily_returns(close)
    atr_pct = (atr(ohlcv["High"], ohlcv["Low"], close) / close).iloc[-1]
    return {
        "ticker": ticker,
        "last_close": close.iloc[-1],
        "rsi_14": rsi(close).iloc[-1],
        "atr_pct": atr_pct,
        "ann_return": annualized_return(returns),
        "ann_vol": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", nargs="+", metavar="TICKER", default=None)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--cache-dir", default=".cache")
    args = parser.parse_args()

    if args.live:
        source = CachedYFinanceSource(args.cache_dir)
        tickers = args.live
    else:
        source = SyntheticSource(seed=1)
        tickers = ["SYNTH_A", "SYNTH_B", "SYNTH_C", "SYNTH_D"]

    rows = []
    for ticker in tickers:
        try:
            ohlcv = source.fetch(ticker, start=args.start)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"skipping {ticker}: {exc}")
            continue
        rows.append(analyze_ticker(ticker, ohlcv))

    report = pd.DataFrame(rows).set_index("ticker").sort_values("sharpe", ascending=False)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(report)


if __name__ == "__main__":
    main()
