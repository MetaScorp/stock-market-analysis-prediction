# Stock Market Analysis and Prediction

A toolkit for stock market analysis - data access, technical and volume
indicators, statistics, risk (VaR/GARCH), regime detection, predictive
modeling, backtesting, and significance testing.

This is a rewrite of an old project of mine from 2016 (dug up again for a
college assignment in 2021, then left alone until now). The old code is
still in `legacy/`, untouched, but it doesn't run anymore - it depended on
Google Finance through `pandas_datareader`, and that backend has been dead
for years. Everything else in this repo is the rewrite, built up in stages
as I had time for it.

## What's here

- **Data** - pluggable sources: `yfinance` for live data (with a local CSV
  cache so you're not re-hitting the network every run), cached CSVs, or
  seeded synthetic data for offline/reproducible work; a data-quality
  checker that flags gaps, stale prices, zero volume, inverted OHLC ranges,
  and outsized single-day moves
- **Indicators** - SMA, EMA, RSI, MACD, Bollinger Bands, OBV, VWAP, ATR,
  Stochastic Oscillator
- **Stats** - returns, annualized return/vol, Sharpe/Sortino/Calmar ratios,
  drawdown series & duration, win rate, profit factor, correlation
- **Diagnostics** - stationarity (ADF), autocorrelation (Ljung-Box), and
  normality (Jarque-Bera) tests, so model/VaR assumptions get checked
  instead of assumed
- **Portfolio** - rolling correlation, rolling & static beta vs. a benchmark,
  relative strength
- **Risk** - historical VaR, parametric VaR, seeded Monte Carlo VaR/CVaR, and
  GARCH(1,1)-based volatility forecasting + VaR for when volatility is
  clearly clustering rather than roughly constant
- **Regime** - rolling volatility regime classification (low/medium/high),
  plus a Markov regime-switching model that estimates regimes and their
  transition probabilities directly from the data
- **Modeling** - lagged features, walk-forward validation, and four models
  to compare: naive persistence baseline, linear regression, gradient
  boosted trees, and ARIMA
- **Backtesting** - long/flat or volatility-targeted position sizing, a
  one-day execution lag, optional transaction costs, and a full metrics set
  (Sharpe/Sortino/Calmar/drawdown/win rate/profit factor), always compared
  against buy-and-hold
- **Evaluation** - bootstrap confidence intervals on Sharpe ratio, a
  permutation test for whether a signal's backtest result beats randomly
  picking the same number of days, and a paired test for comparing two
  forecasting models properly instead of eyeballing two RMSE numbers
- **Viz** - matplotlib plots that return `Figure` objects instead of calling
  `plt.show()`, so they're actually testable
- Tests for all of the above, fully offline (no network calls), running in
  CI on 3.10/3.11/3.12

## Install

Needs Python 3.10+.

```bash
git clone <this-repo-url>
cd stock-analysis-platform
pip install -e ".[dev]"         # core + test/lint tools
pip install -e ".[live]"        # optional, adds yfinance for real data
pip install -e ".[research]"    # optional, adds statsmodels + arch (GARCH, ADF, Markov switching, ARIMA)
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
python examples/05_volatility_and_regimes.py    # diagnostics, GARCH, regimes, vol targeting
python examples/06_research_report.py           # full pipeline -> one report, with caveats
python examples/07_real_data_case_study.py      # same pipeline, real 1987 AAPL data incl. Black Monday
```

All of them run offline against synthetic data by default. Pass `--live
TICKER` (or `--live TICKER1 TICKER2 ...` for the screener) to pull real
data - needs the `live` extra, network access, and `05`/`06` also need the
`research` extra installed.

## Layout

```
src/stockanalysis/
├── config.py       # default settings, incl. random seed
├── data/           # pluggable data sources + quality checks
├── indicators/     # technical + volume indicators
├── stats/          # returns, performance metrics, diagnostics
├── portfolio.py    # cross-ticker correlation / beta / relative strength
├── risk/           # VaR / CVaR / GARCH
├── regime/         # volatility regime classification + Markov switching
├── models/         # forecasting + walk-forward validation
├── backtest/       # backtesting engine + position sizing
├── evaluation/     # bootstrap CI, permutation testing, model comparison
└── viz/            # plotting functions

tests/        # pytest suite, offline
examples/     # runnable scripts
legacy/       # the original 2016 project, unmodified
```

## Tests

```bash
pytest                              # run the suite
pytest --cov=stockanalysis          # with coverage
ruff check src tests                # lint
```

Everything runs against a seeded `SyntheticSource`, so there's no network
dependency and results are the same locally and in CI. `pip install -e
".[dev,research]"` first, or the diagnostics/GARCH/Markov/ARIMA tests will
fail on import.

## Reproducibility

Anything involving randomness (synthetic data, Monte Carlo, bootstrap,
permutation tests) takes an explicit seed and defaults to a fixed one.
Same inputs, same seed, same numbers every time.

## On the forecasts

Nothing in here predicts the market. The forecasting models are evaluated
with walk-forward validation against a naive persistence baseline, and
`evaluation/` exists specifically to check whether a backtest's edge is
distinguishable from noise before getting excited about it. A model that
"beats the baseline" on one historical window, on one ticker, is a
hypothesis worth testing further - not a result to trade on. Every number
this project produces is meant to be reproducible and challengeable, not
taken on faith.

## Why not pandas_datareader anymore

The old project used `pandas_datareader`'s Google Finance backend, which
Google killed off and `pandas_datareader` formally deprecated a while back.
`yfinance` is the current standard for free historical data in Python, so
that's what `YFinanceSource` uses now. It sits behind the same `DataSource`
interface as everything else, so if it ever meets the same fate, only one
file needs to change.

## Real data, not just synthetic

Most of the test suite runs against `SyntheticSource` because it's fast,
offline, and deterministic - but it's i.i.d. by construction, so it never
shows fat tails, real autocorrelation, or an actual crash.
`examples/data/AAPL_1987.csv` bundles real, unmodified daily AAPL data for
1987 (Black Monday included), and `07_real_data_case_study.py` plus
`tests/test_real_data_integration.py` run the full pipeline against it.
The diagnostics genuinely behave differently there - normality and
no-autocorrelation both get rejected, which never happens on synthetic
data. See `examples/data/README.md` for where it came from.

`YFinanceSource`/`CachedYFinanceSource` are real, working code, but I
haven't been able to exercise a live fetch from every environment I've
run this in - some sandboxes block outbound requests to Yahoo Finance
specifically while still allowing PyPI/GitHub. If `--live` fails with a
network/host error rather than a data error, that's what's happening.

## How this was built

Built in four rough passes rather than one shot:

1. Data loading, basic indicators (SMA/EMA/RSI/MACD/Bollinger), return
   stats, a simple Monte Carlo VaR, one backtest.
2. Volume indicators, a full performance-metrics set (Sortino/Calmar/
   drawdown duration/win rate/profit factor), a caching data source,
   cross-ticker correlation/beta, a multi-ticker screener.
3. Stationarity/autocorrelation/normality tests, GARCH volatility
   forecasting, rolling volatility regimes, volatility-targeted position
   sizing, gradient boosting + ARIMA models.
4. Markov regime-switching, bootstrap/permutation significance testing,
   a paired test for comparing two forecasting models properly, data
   quality checks, and the real-1987-data case study.

## License

MIT - see `LICENSE`.
