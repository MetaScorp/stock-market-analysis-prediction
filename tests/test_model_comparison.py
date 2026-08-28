import numpy as np
import pytest

from stockanalysis.evaluation import compare_models
from stockanalysis.models import build_lagged_features, walk_forward_validate
from stockanalysis.models.forecasting import FoldResult


def _fold(seed: int, n: int, scale: float) -> FoldResult:
    rng = np.random.default_rng(seed)
    errors = rng.chisquare(df=2, size=n) * scale  # always positive, like squared errors
    return FoldResult(fold=1, n_train=100, n_test=n, rmse=float(np.sqrt(errors.mean())),
                       mae=0.0, squared_errors=errors)


def test_identical_errors_show_no_significant_difference():
    fold = _fold(seed=0, n=200, scale=1.0)
    result = compare_models([fold], [fold])
    assert not result.a_is_significantly_better
    assert not result.b_is_significantly_better
    assert result.mean_error_diff == pytest.approx(0.0)


def test_clearly_worse_model_a_is_detected():
    fold_a = _fold(seed=0, n=500, scale=3.0)  # much bigger errors
    fold_b = _fold(seed=0, n=500, scale=1.0)
    result = compare_models([fold_a], [fold_b])
    assert result.b_is_significantly_better
    assert not result.a_is_significantly_better


def test_clearly_worse_model_b_is_detected():
    fold_a = _fold(seed=0, n=500, scale=1.0)
    fold_b = _fold(seed=0, n=500, scale=3.0)
    result = compare_models([fold_a], [fold_b])
    assert result.a_is_significantly_better
    assert not result.b_is_significantly_better


def test_mismatched_fold_counts_raise():
    fold = _fold(seed=0, n=100, scale=1.0)
    with pytest.raises(ValueError):
        compare_models([fold, fold], [fold])


def test_mismatched_fold_sizes_raise():
    fold_a = _fold(seed=0, n=100, scale=1.0)
    fold_b = _fold(seed=0, n=150, scale=1.0)
    with pytest.raises(ValueError):
        compare_models([fold_a], [fold_b])


def test_reproducible_with_same_seed():
    fold_a = _fold(seed=1, n=300, scale=1.5)
    fold_b = _fold(seed=2, n=300, scale=1.0)
    r1 = compare_models([fold_a], [fold_b], seed=7)
    r2 = compare_models([fold_a], [fold_b], seed=7)
    assert r1.ci_lower == r2.ci_lower
    assert r1.ci_upper == r2.ci_upper


def test_integrates_with_real_walk_forward_folds(close):
    from stockanalysis.models import BaselinePersistenceModel, LinearReturnModel
    from stockanalysis.stats import daily_returns

    returns = daily_returns(close)
    X, y = build_lagged_features(returns, lags=5)
    baseline = walk_forward_validate(BaselinePersistenceModel, X, y, n_splits=4)
    linear = walk_forward_validate(LinearReturnModel, X, y, n_splits=4)

    result = compare_models(baseline, linear)
    assert result.n_observations == sum(f.n_test for f in baseline)
