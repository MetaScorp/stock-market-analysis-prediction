"""
Markov regime-switching volatility model, via statsmodels.

classify_volatility_regime (volatility_regime.py) buckets days by where
rolling realized vol sits in its own history - simple, but the boundaries
are arbitrary quantile cuts and it can't tell you how persistent a regime
is expected to be. A Markov-switching model estimates the regimes and
their transition probabilities directly from the data: it finds the
mean/variance of each state and how likely the series is to stay in one
regime vs. jump to the other, via maximum likelihood.

This is slower and can fail to converge on short or unusual series -
that's real, not a bug to hide. Check `converged` before trusting the
output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MarkovRegimeResult:
    regime: pd.Series  # most likely regime at each date (0 = lower-variance)
    regime_probabilities: pd.DataFrame  # smoothed P(state) per date, one col/state
    regime_means: np.ndarray
    regime_std: np.ndarray
    transition_matrix: np.ndarray  # [i, j] = P(state j at t | state i at t-1)
    converged: bool

    def expected_duration(self) -> np.ndarray:
        """Expected number of periods spent in each regime once entered:
        1 / (1 - P(stay)). A high self-transition probability means a
        regime that persists for a long time once it starts."""
        stay_prob = np.diag(self.transition_matrix)
        return 1 / (1 - stay_prob)

    def summary(self) -> str:
        lines = [f"Converged: {self.converged}"]
        order = np.argsort(self.regime_std)
        for rank, state in enumerate(order):
            label = "low-vol" if rank == 0 else "high-vol"
            lines.append(
                f"  regime {state} ({label}): mean={self.regime_means[state]:.4%}, "
                f"std={self.regime_std[state]:.4%}, "
                f"expected duration={self.expected_duration()[state]:.1f} periods"
            )
        return "\n".join(lines)


def _empty_result(returns: pd.Series, k_regimes: int) -> MarkovRegimeResult:
    """Placeholder returned when the optimizer fails outright (not just
    fails to converge cleanly - an actual numerical error). converged=False
    either way, so callers checking that flag are safe regardless."""
    nan_regime = pd.Series(np.nan, index=returns.index, name="regime")
    nan_probs = pd.DataFrame(
        np.nan, index=returns.index, columns=[f"P(regime {i})" for i in range(k_regimes)]
    )
    return MarkovRegimeResult(
        regime=nan_regime,
        regime_probabilities=nan_probs,
        regime_means=np.full(k_regimes, np.nan),
        regime_std=np.full(k_regimes, np.nan),
        transition_matrix=np.full((k_regimes, k_regimes), np.nan),
        converged=False,
    )


def fit_markov_switching(
    returns: pd.Series, k_regimes: int = 2, seed: int = 42
) -> MarkovRegimeResult:
    """Fit a k-regime Markov-switching model with regime-specific mean and
    variance on a return series. Two regimes (calm/turbulent) is the
    standard starting point; more regimes need more data to identify
    reliably.

    The random search over starting parameters (search_reps) is seeded -
    without that, `fit()` uses the global numpy RNG and this function
    would return slightly different regimes on every call. The optimizer
    can still fail outright on some series (a genuine numerical issue with
    this kind of MLE, not something specific to this code) - that case
    comes back as converged=False with NaN fields instead of a crash, same
    as a normal non-convergence."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    returns = returns.dropna()
    model = MarkovRegression(
        returns, k_regimes=k_regimes, trend="c", switching_variance=True
    )
    rng = np.random.default_rng(seed)
    try:
        fitted = model.fit(search_reps=20, rng=rng)
    except (np.linalg.LinAlgError, ValueError):
        return _empty_result(returns, k_regimes)

    probs = fitted.smoothed_marginal_probabilities
    probs.columns = [f"P(regime {i})" for i in range(k_regimes)]
    most_likely = probs.to_numpy().argmax(axis=1)

    means = np.array([fitted.params[f"const[{i}]"] for i in range(k_regimes)])
    variances = np.array([fitted.params[f"sigma2[{i}]"] for i in range(k_regimes)])

    transition = np.array(
        [
            [fitted.params.get(f"p[{i}->{j}]", np.nan) for j in range(k_regimes)]
            for i in range(k_regimes)
        ]
    )
    # statsmodels only reports k-1 transition probs per row, fill the rest
    for i in range(k_regimes):
        known = transition[i, ~np.isnan(transition[i])]
        if np.isnan(transition[i]).any():
            transition[i, np.isnan(transition[i])] = 1 - known.sum()

    return MarkovRegimeResult(
        regime=pd.Series(most_likely, index=returns.index, name="regime"),
        regime_probabilities=probs,
        regime_means=means,
        regime_std=np.sqrt(variances),
        transition_matrix=transition,
        converged=bool(fitted.mle_retvals.get("converged", False)),
    )
