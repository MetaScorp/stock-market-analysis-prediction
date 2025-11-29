"""
Quickstart: fetch data, compute indicators, print some stats.

Runs offline with synthetic data by default:

    python examples/01_quickstart.py

Or pull real data:

    python examples/01_quickstart.py --live AAPL
"""

from __future__ import annotations

import argparse

from stockanalysis.data import SyntheticSource, YFinanceSource
from stockanalysis.indicators import bollinger_bands, macd, rsi, sma
from stockanalysis.stats import (
    annualized_return,
    annualized_volatility,
    daily_returns,
    sharpe_ratio,
)
from stockanalysis.viz import plot_price_with_bands, plot_returns_distribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", metavar="TICKER",
        help="Fetch a real ticker via yfinance instead of synthetic data.",
    )
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()

    if args.live:
        source = YFinanceSource()
        ticker = args.live
    else:
        source = SyntheticSource(seed=42)
        ticker = "SYNTH"

    ohlcv = source.fetch(ticker, start=args.start, end=args.end)
    close = ohlcv["Close"]
    print(f"Loaded {len(close)} rows for {ticker} ({close.index[0].date()} .. {close.index[-1].date()})")

    returns = daily_returns(close)
    print(f"\nReturn statistics:")
    print(f"  Annualized return:     {annualized_return(returns):.2%}")
    print(f"  Annualized volatility: {annualized_volatility(returns):.2%}")
    print(f"  Sharpe ratio:          {sharpe_ratio(returns):.2f}")

    bands = bollinger_bands(close)
    rsi_values = rsi(close)
    macd_df = macd(close)
    print(f"\nLatest RSI(14): {rsi_values.iloc[-1]:.1f}")
    print(f"Latest MACD histogram: {macd_df['histogram'].iloc[-1]:.4f}")

    fig1 = plot_price_with_bands(close, bands, title=f"{ticker} - Price & Bollinger Bands")
    fig2 = plot_returns_distribution(returns)

    if args.save_plots:
        fig1.savefig("price_bands.png", dpi=150)
        fig2.savefig("returns_distribution.png", dpi=150)
        print("\nSaved price_bands.png and returns_distribution.png")


if __name__ == "__main__":
    main()
