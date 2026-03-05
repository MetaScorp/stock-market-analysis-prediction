"""
GARCH(1,1) volatility forecasting, via the `arch` package.

historical/parametric/Monte Carlo VaR (risk.var) all use one volatility
number for the whole window. That's fine when volatility is roughly
constant, but it isn't - it clusters (calm periods followed by calm
periods, turbulent followed by turbulent). GARCH models that directly:
today's variance depends on yesterday's variance and yesterday's shock.
This gives VaR that reacts to current conditions instead of lagging
behind a long-window average.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GarchFitResult:
    params: dict
    conditional_volatility: pd.Series  # in-sample fitted daily vol (%, not annualized)
    forecast_volatility: np.ndarray  # daily vol forecast for each horizon step

    def annualized_forecast(self, periods_per_year: int = 252) -> np.ndarray:
        return self.forecast_volatility * np.sqrt(periods_per_year)


def fit_garch(
    returns: pd.Series, horizon_days: int = 5, p: int = 1, q: int = 1
) -> GarchFitResult:
    """Fit a GARCH(p, q) model with a constant mean, normal errors.

    `arch` works best with returns scaled to roughly unit variance (percent,
    not fraction) - fitting on raw fractional returns (~0.01) can fail to
    converge, so returns are rescaled internally and the results converted
    back before being handed back.
    """
    from arch import arch_model

    returns_pct = returns.dropna() * 100
    model = arch_model(returns_pct, mean="Constant", vol="GARCH", p=p, q=q, dist="normal")
    fitted = model.fit(disp="off")

    forecast = fitted.forecast(horizon=horizon_days, reindex=False)
    forecast_vol_pct = np.sqrt(forecast.variance.values[-1])  # daily vol, in % units

    return GarchFitResult(
        params=fitted.params.to_dict(),
        conditional_volatility=(fitted.conditional_volatility / 100).rename("garch_vol"),
        forecast_volatility=forecast_vol_pct / 100,
    )


def garch_var(
    returns: pd.Series, confidence: float = 0.95, horizon_days: int = 1
) -> float:
    """1-step-ahead parametric VaR using GARCH's forecast volatility instead
    of the full-sample historical std. Assumes normal errors, same as
    risk.parametric_var, so the two are directly comparable - the
    difference isolates the effect of using a time-varying vol estimate."""
    from scipy.stats import norm

    result = fit_garch(returns, horizon_days=horizon_days)
    mu = returns.mean()
    sigma = result.forecast_volatility[-1]
    z = norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))
