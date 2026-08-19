import pandas as pd
import pytest

from stockanalysis.data import SyntheticSource, check_data_quality


def test_clean_synthetic_data_has_no_issues():
    df = SyntheticSource(seed=1).fetch("X", "2020-01-01", "2020-06-01")
    report = check_data_quality(df)
    assert report.is_clean
    assert report.n_rows == len(df)


def test_detects_missing_business_days():
    df = SyntheticSource(seed=1).fetch("X", "2020-01-01", "2020-03-01")
    gapped = df.drop(df.index[10])
    report = check_data_quality(gapped)
    assert len(report.missing_business_days) == 1
    assert report.missing_business_days[0] == df.index[10]


def test_detects_stale_prices():
    df = SyntheticSource(seed=1).fetch("X", "2020-01-01", "2020-03-01").copy()
    df.iloc[5, df.columns.get_loc("Close")] = df.iloc[4]["Close"]
    report = check_data_quality(df)
    assert df.index[5] in report.stale_price_dates


def test_detects_non_positive_prices():
    df = SyntheticSource(seed=1).fetch("X", "2020-01-01", "2020-03-01").copy()
    df.iloc[3, df.columns.get_loc("Close")] = 0.0
    report = check_data_quality(df)
    assert df.index[3] in report.non_positive_price_dates


def test_detects_zero_volume():
    df = SyntheticSource(seed=1).fetch("X", "2020-01-01", "2020-03-01").copy()
    df.iloc[7, df.columns.get_loc("Volume")] = 0
    report = check_data_quality(df)
    assert df.index[7] in report.zero_volume_dates


def test_detects_extreme_moves():
    df = SyntheticSource(seed=1).fetch("X", "2020-01-01", "2020-03-01").copy()
    df.iloc[9, df.columns.get_loc("Close")] = df.iloc[8]["Close"] * 2  # +100% day
    report = check_data_quality(df, extreme_move_threshold=0.20)
    assert df.index[9] in report.extreme_move_dates


def test_detects_inverted_high_low():
    df = SyntheticSource(seed=1).fetch("X", "2020-01-01", "2020-03-01").copy()
    df.iloc[6, df.columns.get_loc("High")] = df.iloc[6]["Low"] - 1
    report = check_data_quality(df)
    assert df.index[6] in report.inverted_range_dates


def test_summary_lists_each_issue_type():
    df = SyntheticSource(seed=1).fetch("X", "2020-01-01", "2020-03-01").copy()
    df.iloc[3, df.columns.get_loc("Volume")] = 0
    text = check_data_quality(df).summary()
    assert "zero volume" in text


def test_raises_on_missing_required_columns():
    df = pd.DataFrame({"Close": [1.0, 2.0]})
    with pytest.raises(ValueError):
        check_data_quality(df)
