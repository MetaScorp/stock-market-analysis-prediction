"""
Value at Risk and CVaR (expected shortfall).

The old project's Monte Carlo VaR ran an unseeded loop-per-path simulation,
so the number changed every time you ran the notebook. This keeps the same
GBM approach but vectorizes it and seeds it, and adds historical/parametric
VaR alongside it so you have something to sanity check against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Empirical VaR from the historical quantile. Positive = loss fraction."""
    q = 1 - confidence
    return float(-returns.quantile(q))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Variance-covariance VaR, assumes normal returns."""
    from scipy.stats import norm

    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))


@dataclass
class MonteCarloResult:
    var: float
    cvar: float
    simulated_returns: np.ndarray
    seed: int
    n_simulations: int
    horizon_days: int

    def summary(self) -> str:
        return (
            f"Monte Carlo VaR ({self.n_simulations:,} sims, "
            f"{self.horizon_days}-day horizon, seed={self.seed}): "
            f"VaR={self.var:.2%}, CVaR={self.cvar:.2%}"
        )


def monte_carlo_var(
    mu: float,
    sigma: float,
    horizon_days: int = 1,
    confidence: float = 0.95,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> MonteCarloResult:
    """Simulate GBM return paths and pull VaR/CVaR from the outcomes."""
    rng = np.random.default_rng(seed)
    drift = (mu - 0.5 * sigma**2) * horizon_days
    diffusion = sigma * np.sqrt(horizon_days) * rng.standard_normal(n_simulations)
    simulated_returns = np.exp(drift + diffusion) - 1

    q = 1 - confidence
    var = float(-np.quantile(simulated_returns, q))
    tail = simulated_returns[simulated_returns <= np.quantile(simulated_returns, q)]
    cvar = float(-tail.mean()) if len(tail) else float("nan")

    return MonteCarloResult(
        var=var,
        cvar=cvar,
        simulated_returns=simulated_returns,
        seed=seed,
        n_simulations=n_simulations,
        horizon_days=horizon_days,
    )
