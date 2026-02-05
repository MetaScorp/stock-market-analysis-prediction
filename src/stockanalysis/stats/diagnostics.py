"""
Statistical diagnostics.

Before trusting a mean/vol estimate or fitting a model, it's worth checking
whether the assumptions actually hold. These wrap statsmodels/scipy tests
that come up constantly in this kind of work: is the series stationary, are
returns autocorrelated (they shouldn't be, much, if markets are
roughly efficient), and are they normally distributed (they're usually
not - fat tails are the norm, not the exception).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    reject_null_at_5pct: bool
    null_hypothesis: str

    def summary(self) -> str:
        verdict = "rejected" if self.reject_null_at_5pct else "not rejected"
        return (
            f"{self.name}: statistic={self.statistic:.4f}, p={self.p_value:.4f} "
            f"-> H0 ({self.null_hypothesis}) {verdict} at 5%"
        )


def adf_test(series: pd.Series) -> TestResult:
    """Augmented Dickey-Fuller test. H0: series has a unit root (is
    non-stationary). Raw prices are almost always non-stationary; returns
    almost always are - this is mostly useful as a sanity check before
    feeding something into a model that assumes stationarity."""
    import warnings

    from statsmodels.tsa.stattools import adfuller

    series = series.dropna()
    with warnings.catch_warnings():
        # newer statsmodels wants result_object=, which doesn't exist on
        # older versions - just suppress the deprecation warning instead
        warnings.filterwarnings("ignore", message=r".*adfuller.*", category=FutureWarning)
        statistic, p_value, *_ = adfuller(series, autolag="AIC")
    return TestResult(
        name="Augmented Dickey-Fuller",
        statistic=float(statistic),
        p_value=float(p_value),
        reject_null_at_5pct=bool(p_value < 0.05),
        null_hypothesis="series has a unit root (non-stationary)",
    )


def ljung_box_test(series: pd.Series, lags: int = 10) -> TestResult:
    """Ljung-Box test for autocorrelation. H0: no autocorrelation up to
    `lags`. A rejection in returns is a real signal (something predictable
    in the mean); a rejection in squared returns just means volatility
    clusters, which it always does - that's a case for GARCH, not alpha."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    result = acorr_ljungbox(series.dropna(), lags=[lags], return_df=True)
    statistic = float(result["lb_stat"].iloc[0])
    p_value = float(result["lb_pvalue"].iloc[0])
    return TestResult(
        name=f"Ljung-Box (lags={lags})",
        statistic=statistic,
        p_value=p_value,
        reject_null_at_5pct=bool(p_value < 0.05),
        null_hypothesis="no autocorrelation",
    )


def jarque_bera_test(series: pd.Series) -> TestResult:
    """Jarque-Bera normality test, based on sample skew and kurtosis.
    H0: series is normally distributed. Daily returns almost always reject
    this - fat tails matter for VaR, which is why parametric_var (normal
    assumption) tends to understate risk relative to historical_var."""
    from scipy.stats import jarque_bera

    statistic, p_value = jarque_bera(series.dropna())
    return TestResult(
        name="Jarque-Bera",
        statistic=float(statistic),
        p_value=float(p_value),
        reject_null_at_5pct=bool(p_value < 0.05),
        null_hypothesis="series is normally distributed",
    )
