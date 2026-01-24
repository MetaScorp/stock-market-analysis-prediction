# Stock Market Analysis and Prediction

A toolkit for stock market analysis - data access, technical indicators,
stats, risk (VaR), a bit of predictive modeling, and backtesting.

This is a rewrite of an old project of mine from 2016 (dug up again for a
college assignment in 2021, then left alone until now). The old code is
still in `legacy/`, untouched, but it doesn't run anymore - it depended on
Google Finance through `pandas_datareader`, and that backend has been dead
for years. Everything else in this repo is the rewrite.

## What's here

- **Data** - pluggable sources: `yfinance` for live data (with a local CSV
  cache so you're not re-hitting the network every run), cached CSVs, or
  seeded synthetic data for offline/reproducible work
- **Indicators** - SMA, EMA, RSI, MACD, Bollinger Bands, OBV, VWAP, ATR,
  Stochastic Oscillator
- **Stats** - returns, annualized return/vol, Sharpe/Sortino/Calmar ratios,
  drawdown series & duration, win rate, profit factor, correlation
- **Portfolio** - rolling correlation, rolling & static beta vs. a benchmark,
  relative strength
- **Risk** - historical VaR, parametric VaR, and a seeded Monte Carlo VaR/CVaR
- **Modeling** - lagged features, walk-forward validation, a naive baseline
  vs. a linear model
- **Backtesting** - long/flat backtests with a one-day execution lag and
  optional transaction costs, reporting Sharpe/Sortino/Calmar/drawdown/win
  rate/profit factor, always compared against buy-and-hold
- **Viz** - matplotlib plots that return `Figure` objects instead of calling
  `plt.show()`, so they're actually testable
- Tests for all of the above, fully offline (no network calls), running in
  CI on 3.10/3.11/3.12

## Install

Needs Python 3.10+.

```bash
git clone <this-repo-url>
cd stock-analysis-platform
pip install -e ".[dev]"     # core + test/lint tools
pip install -e ".[live]"    # optional, adds yfinance for real data
```

## Quickstart

```python
from stockanalysis.data import SyntheticSource
from stockanalysis.indicators import bollinger_bands, rsi
from stockanalysis.stats import daily_returns, sharpe_ratio

source = SyntheticSource(seed=42)   # swap for YFinanceSource() for real data
prices = source.fetch("DEMO", start="2020-01-01")["Close"]

returns = daily_returns(prices)
print(f"Sharpe ratio: {sharpe_ratio(returns):.2f}")
print(f"Latest RSI:   {rsi(prices).iloc[-1]:.1f}")

bands = bollinger_bands(prices)
```

There are also a few runnable scripts in `examples/`:

```bash
python examples/01_quickstart.py               # data + indicators + stats
python examples/02_risk_analysis.py             # VaR/CVaR, three methods
python examples/03_modeling_and_backtest.py     # forecasting + backtest
python examples/04_multi_ticker_screener.py     # rank a list of tickers
```

All three run offline against synthetic data by default. Pass
`--live TICKER` to `01_quickstart.py` to pull real data (needs the `live`
extra and network access).

## Layout

```
src/stockanalysis/
├── config.py       # default settings, incl. random seed
├── data/           # pluggable data sources
├── indicators/     # technical indicators
├── stats/          # returns & performance stats
├── risk/           # VaR / CVaR
├── models/         # forecasting + walk-forward validation
├── backtest/       # backtesting engine
└── viz/            # plotting functions

tests/        # pytest suite, offline
examples/     # runnable scripts
legacy/       # the original 2016 project, unmodified
```

## Tests

```bash
pytest                       # run the suite
pytest --cov=stockanalysis   # with coverage
ruff check src tests         # lint
```

Everything runs against a seeded `SyntheticSource`, so there's no network
dependency and results are the same locally and in CI.

## Reproducibility

Anything involving randomness (synthetic data, Monte Carlo) takes an
explicit seed and defaults to a fixed one in `config.DEFAULT_SETTINGS`.
Same inputs, same seed, same numbers every time.

## Why not pandas_datareader anymore

The old project used `pandas_datareader`'s Google Finance backend, which
Google killed off and `pandas_datareader` formally deprecated a while back.
`yfinance` is the current standard for free historical data in Python, so
that's what `YFinanceSource` uses now. It sits behind the same `DataSource`
interface as everything else, so if it ever meets the same fate, only one
file needs to change.

## License

MIT - see `LICENSE`.
