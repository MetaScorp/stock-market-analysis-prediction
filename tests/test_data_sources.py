import pandas as pd
import pytest

from stockanalysis.data import CSVSource, SyntheticSource
from stockanalysis.data.sources import REQUIRED_COLUMNS


def test_synthetic_source_returns_expected_columns():
    df = SyntheticSource(seed=1).fetch("AAA", start="2022-01-01", end="2022-03-01")
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing


def test_synthetic_source_is_reproducible():
    df1 = SyntheticSource(seed=123).fetch("AAA", "2022-01-01", "2022-06-01")
    df2 = SyntheticSource(seed=123).fetch("AAA", "2022-01-01", "2022-06-01")
    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_source_differs_by_ticker():
    src = SyntheticSource(seed=123)
    aaa = src.fetch("AAA", "2022-01-01", "2022-06-01")
    bbb = src.fetch("BBB", "2022-01-01", "2022-06-01")
    assert not aaa["Close"].equals(bbb["Close"])


def test_synthetic_source_rejects_empty_range():
    with pytest.raises(ValueError):
        SyntheticSource().fetch("AAA", "2022-01-05", "2022-01-01")


def test_csv_source_roundtrip(tmp_path):
    original = SyntheticSource(seed=5).fetch("AAA", "2022-01-01", "2022-02-01")
    original.to_csv(tmp_path / "AAA.csv", index_label="Date")

    loaded = CSVSource(tmp_path).fetch("AAA", "2022-01-01", "2022-02-01")
    pd.testing.assert_frame_equal(loaded, original, check_dtype=False)


def test_csv_source_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CSVSource(tmp_path).fetch("NOPE", "2022-01-01", "2022-02-01")
