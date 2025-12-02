import numpy as np
import pandas as pd

from stockanalysis.models import (
    BaselinePersistenceModel,
    LinearReturnModel,
    build_lagged_features,
    train_test_split_ts,
    walk_forward_validate,
)


def test_build_lagged_features_shapes_and_alignment():
    returns = pd.Series(np.arange(10, dtype=float), index=pd.RangeIndex(10))
    X, y = build_lagged_features(returns, lags=3)
    assert list(X.columns) == ["lag_1", "lag_2", "lag_3"]
    assert len(X) == len(y) == 7
    first_y_index = y.index[0]
    assert X.loc[first_y_index, "lag_1"] == returns.loc[first_y_index - 1]


def test_train_test_split_ts_preserves_chronological_order():
    X = pd.DataFrame({"a": range(10)})
    y = pd.Series(range(10))
    X_train, X_test, _, _ = train_test_split_ts(X, y, test_size=0.3)
    assert list(X_train.index) == list(range(7))
    assert list(X_test.index) == list(range(7, 10))
    assert X_train.index.max() < X_test.index.min()


def test_baseline_persistence_model_predicts_lag_1():
    model = BaselinePersistenceModel()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    preds = model.predict(X)
    np.testing.assert_array_equal(preds, [1.0, 3.0])


def test_linear_model_fits_a_perfect_linear_relationship():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1))
    y = 2.0 * X[:, 0] + 1.0
    model = LinearReturnModel().fit(X, y)
    preds = model.predict(X)
    assert np.allclose(preds, y, atol=1e-8)


def test_walk_forward_validate_runs_without_lookahead(close):
    from stockanalysis.stats import daily_returns

    returns = daily_returns(close)
    X, y = build_lagged_features(returns, lags=5)
    results = walk_forward_validate(BaselinePersistenceModel, X, y, n_splits=4)

    assert len(results) == 4
    for fold in results:
        assert fold.rmse >= 0
        assert fold.mae >= 0
        assert fold.n_train > 0 and fold.n_test > 0
