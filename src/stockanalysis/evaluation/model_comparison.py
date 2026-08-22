"""
Is model A actually better than model B, or did it just get a luckier set
of test points?

walk_forward_validate reports RMSE per fold for one model at a time.
Comparing two RMSE numbers directly ("0.0162 < 0.0182, model A wins")
ignores that both numbers came from a small, specific set of test points -
the difference could easily be noise. This runs both models over the same
walk-forward folds (same train/test splits, so the comparison is paired -
each test point contributes one squared-error difference) and bootstraps
a confidence interval on the mean error difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stockanalysis.models.forecasting import FoldResult


@dataclass
class ModelComparisonResult:
    model_a_rmse: float
    model_b_rmse: float
    mean_error_diff: float  # mean(squared_error_a - squared_error_b); >0 means A is worse
    ci_lower: float
    ci_upper: float
    confidence: float
    n_observations: int

    @property
    def b_is_significantly_better(self) -> bool:
        """True only if the whole CI on (error_a - error_b) is above zero -
        i.e. A being worse than B isn't just consistent with zero difference."""
        return self.ci_lower > 0

    @property
    def a_is_significantly_better(self) -> bool:
        return self.ci_upper < 0

    def summary(self) -> str:
        if self.a_is_significantly_better:
            verdict = "model A is significantly better"
        elif self.b_is_significantly_better:
            verdict = "model B is significantly better"
        else:
            verdict = "no significant difference"
        return (
            f"RMSE A={self.model_a_rmse:.5f}, B={self.model_b_rmse:.5f} | "
            f"mean squared-error diff (A-B): {self.mean_error_diff:.6f} "
            f"[{self.confidence:.0%} CI: {self.ci_lower:.6f}, {self.ci_upper:.6f}] "
            f"-> {verdict} on {self.n_observations} paired observations"
        )


def compare_models(
    fold_results_a: list[FoldResult],
    fold_results_b: list[FoldResult],
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> ModelComparisonResult:
    """fold_results_a/b must come from walk_forward_validate calls on the
    same X/y with the same n_splits, so the test points line up fold by
    fold. Concatenates each fold's squared errors and paired-bootstraps
    the mean difference."""
    if len(fold_results_a) != len(fold_results_b):
        raise ValueError("Both model runs must have the same number of folds.")

    errors_a = np.concatenate([f.squared_errors for f in fold_results_a])
    errors_b = np.concatenate([f.squared_errors for f in fold_results_b])
    if len(errors_a) != len(errors_b):
        raise ValueError(
            "Fold sizes differ between the two runs - were they run on the "
            "same X/y with the same n_splits?"
        )

    diff = errors_a - errors_b
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(diff, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1 - confidence
    lower, upper = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])

    return ModelComparisonResult(
        model_a_rmse=float(np.sqrt(errors_a.mean())),
        model_b_rmse=float(np.sqrt(errors_b.mean())),
        mean_error_diff=float(diff.mean()),
        ci_lower=float(lower),
        ci_upper=float(upper),
        confidence=confidence,
        n_observations=n,
    )
