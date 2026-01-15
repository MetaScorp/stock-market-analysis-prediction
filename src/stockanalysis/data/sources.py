"""
Market data sources.

The old project called pandas_datareader's Google Finance backend directly,
inline, everywhere it needed prices. When that backend died, the whole
project died with it. So this time data access is behind one interface
(DataSource) and everything downstream just consumes a plain DataFrame.
Swap the source, nothing else has to change.
"""

from __future__ import annotations

import abc
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataSource(abc.ABC):
    """Every data source implements this."""

    @abc.abstractmethod
    def fetch(self, ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
        """OHLCV data for ticker between start and end, indexed by date."""
        raise NotImplementedError

    def fetch_many(
        self, tickers: list[str], start: str, end: str | None = None
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            out[ticker] = self.fetch(ticker, start, end)
        return out

    @staticmethod
    def _validate(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Data for {ticker!r} is missing columns {missing}. "
                f"Got: {list(df.columns)}"
            )
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"Data for {ticker!r} must have a DatetimeIndex.")
        return df.sort_index()


class YFinanceSource(DataSource):
    """Live data via yfinance. Replaces the old Google Finance call."""

    def __init__(self) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "YFinanceSource requires the 'yfinance' package. "
                "Install it with: pip install yfinance"
            ) from exc

    def fetch(self, ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(
            ticker, start=start, end=end, progress=False, auto_adjust=False
        )
        if df.empty:
            raise ValueError(
                f"No data returned for {ticker!r} in range {start}..{end}. "
                "Check the ticker symbol and date range."
            )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return self._validate(df[REQUIRED_COLUMNS], ticker)


class CSVSource(DataSource):
    """Load previously-saved OHLCV data from local CSVs, one file per ticker."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def fetch(self, ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
        path = self.directory / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No cached CSV for {ticker!r} at {path}")
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        df.index.name = None
        df = df.loc[start:end]
        df.index.freq = pd.infer_freq(df.index)  # lost on the CSV round-trip
        return self._validate(df, ticker)


class CachedYFinanceSource(DataSource):
    """Wraps another DataSource (YFinanceSource by default) and writes each
    fetch to a local CSV, reading from that cache on later calls so
    re-running an analysis doesn't re-hit the network."""

    def __init__(
        self,
        directory: str | Path,
        live_source: DataSource | None = None,
        refresh: bool = False,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.refresh = refresh
        self._live = live_source if live_source is not None else YFinanceSource()
        self._cache = CSVSource(self.directory)

    def fetch(self, ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
        cache_path = self.directory / f"{ticker}.csv"
        if cache_path.exists() and not self.refresh and self._cache_covers(
            cache_path, start, end
        ):
            return self._cache.fetch(ticker, start, end)
        df = self._live.fetch(ticker, start, end)
        df.to_csv(cache_path, index_label="Date")
        return df

    @staticmethod
    def _cache_covers(cache_path: Path, start: str, end: str | None) -> bool:
        """Cheap check that the cached file's date range covers the request,
        so we don't silently hand back a partial slice from a narrower fetch."""
        dates = pd.read_csv(cache_path, usecols=["Date"], parse_dates=["Date"])["Date"]
        if dates.empty:
            return False
        requested_end = pd.Timestamp(end) if end else pd.Timestamp.today()
        return dates.min() <= pd.Timestamp(start) and dates.max() >= requested_end - pd.Timedelta(days=5)


class SyntheticSource(DataSource):
    """Seeded synthetic OHLCV (geometric Brownian motion), no network needed.

    Used by the test suite and examples so they run the same everywhere.
    """

    def __init__(
        self,
        seed: int = 42,
        annual_drift: float = 0.08,
        annual_volatility: float = 0.25,
        start_price: float = 100.0,
    ) -> None:
        self.seed = seed
        self.annual_drift = annual_drift
        self.annual_volatility = annual_volatility
        self.start_price = start_price

    def fetch(self, ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        dates = pd.bdate_range(start=start, end=end)
        n = len(dates)
        if n == 0:
            raise ValueError(f"Empty date range: {start}..{end}")

        # derive the seed from the ticker so different symbols get
        # different (still reproducible) paths from one base seed
        rng = np.random.default_rng(self.seed + sum(map(ord, ticker)))

        dt = 1 / 252
        mu, sigma = self.annual_drift, self.annual_volatility
        shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
        close = self.start_price * np.exp(np.cumsum(shocks))

        daily_range = np.abs(rng.normal(0, sigma * np.sqrt(dt), n)) * close
        high = close + daily_range
        low = np.maximum(close - daily_range, 0.01)
        open_ = np.concatenate([[self.start_price], close[:-1]])
        volume = rng.integers(1_000_000, 5_000_000, n)

        df = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=dates,
        )
        return self._validate(df, ticker)
