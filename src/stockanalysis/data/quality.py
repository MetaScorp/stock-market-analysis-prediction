"""
Data quality checks.

Real market data has problems the synthetic generator never will: missing
trading days, stale prices repeated because a feed didn't update, zero or
negative values that shouldn't be possible, and single-day moves so large
they're more likely a data error (unadjusted split) than a real move.
None of this fixes the data - it just tells you what's wrong so you can
decide whether to trust it, before an indicator or backtest quietly runs on
garbage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class DataQualityReport:
    n_rows: int
    missing_business_days: pd.DatetimeIndex
    stale_price_dates: pd.DatetimeIndex  # Close unchanged from the prior row
    non_positive_price_dates: pd.DatetimeIndex
    zero_volume_dates: pd.DatetimeIndex
    extreme_move_dates: pd.DatetimeIndex  # |daily return| above the threshold
    inverted_range_dates: pd.DatetimeIndex  # High < Low, or Close outside [Low, High]
    issues: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0

    def summary(self) -> str:
        if self.is_clean:
            return f"{self.n_rows} rows, no issues found."
        lines = [f"{self.n_rows} rows, {len(self.issues)} issue type(s) found:"]
        lines.extend(f"  - {issue}" for issue in self.issues)
        return "\n".join(lines)


def check_data_quality(
    ohlcv: pd.DataFrame, extreme_move_threshold: float = 0.20
) -> DataQualityReport:
    """Run the checks above and return a report. extreme_move_threshold is
    a fraction (0.20 = 20%) - real stocks do sometimes move this much in a
    day, so treat a hit here as "worth a second look", not "definitely
    wrong"."""
    missing = [c for c in REQUIRED_COLUMNS if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    close = ohlcv["Close"]

    expected_days = pd.bdate_range(ohlcv.index.min(), ohlcv.index.max())
    missing_days = expected_days.difference(ohlcv.index)

    stale = ohlcv.index[(close.diff() == 0).fillna(False)]

    non_positive = ohlcv.index[(ohlcv[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)]

    zero_volume = ohlcv.index[ohlcv["Volume"] == 0]

    daily_return = close.pct_change()
    extreme_moves = ohlcv.index[daily_return.abs() > extreme_move_threshold]

    inverted = ohlcv.index[
        (ohlcv["High"] < ohlcv["Low"])
        | (ohlcv["Close"] > ohlcv["High"])
        | (ohlcv["Close"] < ohlcv["Low"])
    ]

    issues = []
    if len(missing_days) > 0:
        issues.append(f"{len(missing_days)} business day(s) missing from the index")
    if len(stale) > 0:
        issues.append(f"{len(stale)} day(s) with an unchanged close (possible stale feed)")
    if len(non_positive) > 0:
        issues.append(f"{len(non_positive)} day(s) with a non-positive OHLC price")
    if len(zero_volume) > 0:
        issues.append(f"{len(zero_volume)} day(s) with zero volume")
    if len(extreme_moves) > 0:
        issues.append(
            f"{len(extreme_moves)} day(s) with a >{extreme_move_threshold:.0%} "
            "single-day move"
        )
    if len(inverted) > 0:
        issues.append(f"{len(inverted)} day(s) with an inconsistent High/Low/Close")

    return DataQualityReport(
        n_rows=len(ohlcv),
        missing_business_days=missing_days,
        stale_price_dates=stale,
        non_positive_price_dates=non_positive,
        zero_volume_dates=zero_volume,
        extreme_move_dates=extreme_moves,
        inverted_range_dates=inverted,
        issues=issues,
    )
