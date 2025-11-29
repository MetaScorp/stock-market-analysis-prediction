"""
Risk assessment: historical, parametric, and Monte Carlo VaR/CVaR.

    python examples/02_risk_analysis.py
"""

from __future__ import annotations

from stockanalysis.config import DEFAULT_SETTINGS
from stockanalysis.data import SyntheticSource
from stockanalysis.risk import historical_var, monte_carlo_var, parametric_var
from stockanalysis.stats import daily_returns
from stockanalysis.viz import plot_var_distribution


def main() -> None:
    source = SyntheticSource(seed=DEFAULT_SETTINGS.random_seed)
    ohlcv = source.fetch("SYNTH", start=DEFAULT_SETTINGS.start_date)
    returns = daily_returns(ohlcv["Close"])

    confidence = DEFAULT_SETTINGS.var_confidence
    mu, sigma = returns.mean(), returns.std(ddof=1)

    hist = historical_var(returns, confidence=confidence)
    param = parametric_var(returns, confidence=confidence)
    mc = monte_carlo_var(
        mu=mu,
        sigma=sigma,
        horizon_days=1,
        confidence=confidence,
        n_simulations=DEFAULT_SETTINGS.mc_simulations,
        seed=DEFAULT_SETTINGS.random_seed,
    )

    print(f"1-day VaR at {confidence:.0%} confidence:")
    print(f"  Historical VaR:  {hist:.2%}")
    print(f"  Parametric VaR:  {param:.2%}")
    print(f"  Monte Carlo VaR: {mc.var:.2%}  (CVaR: {mc.cvar:.2%})")
    print(f"\n{mc.summary()}")

    fig = plot_var_distribution(mc.simulated_returns, mc.var, confidence)
    fig.savefig("var_distribution.png", dpi=150)
    print("\nSaved var_distribution.png")


if __name__ == "__main__":
    main()
