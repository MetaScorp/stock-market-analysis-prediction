"""Shared fixtures. Everything runs against SyntheticSource so tests are
offline and deterministic."""

import pandas as pd
import pytest

from stockanalysis.data import SyntheticSource


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    source = SyntheticSource(seed=7)
    return source.fetch("TEST", start="2020-01-01", end="2021-12-31")


@pytest.fixture
def close(ohlcv) -> pd.Series:
    return ohlcv["Close"]
