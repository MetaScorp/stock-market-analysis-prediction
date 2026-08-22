"""
Next-day return forecasting.

The original project's "prediction" was really just simulating future price
paths from historical mean/vol (the Monte Carlo bit in risk/var.py), not an
actual model fit and evaluated on held-out data. This adds that: lagged
features, a naive baseline, a linear model, and walk-forward validation so
we're not accidentally leaking future data into training (an easy mistake
with time series and a random train_test_split).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


class Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Model: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def build_lagged_features(
    returns: pd.Series, lags: int = 5
) -> tuple[pd.DataFrame, pd.Series]:
    """Lagged-return feature matrix + next-day-return target.

    X row at date t holds returns for t-1 .. t-lags. y at t is the return
    realized on t, predicted using only information strictly before it.
    """
    df = pd.DataFrame({"y": returns})
    feature_cols = []
    for lag in range(1, lags + 1):
        col = f"lag_{lag}"
        df[col] = returns.shift(lag)
        feature_cols.append(col)
    df = df.dropna()
    return df[feature_cols], df["y"]


def train_test_split_ts(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronological (non-shuffled) train/test split, by position."""
    n_test = max(1, int(len(X) * test_size))
    split = len(X) - n_test
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


class BaselinePersistenceModel:
    """Predict tomorrow == today (lag_1). Anything fancier should beat this."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaselinePersistenceModel:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return X[:, 0]  # lag_1 column


class LinearReturnModel:
    """Thin wrapper around sklearn's LinearRegression."""

    def __init__(self) -> None:
        from sklearn.linear_model import LinearRegression

        self._model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearReturnModel:
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class GradientBoostingReturnModel:
    """sklearn's gradient boosted trees. Captures non-linear interactions
    between lags that a plain linear model can't - at the cost of being
    much easier to overfit with only a handful of lag features, so keep
    an eye on whether it's actually beating the linear model out of
    sample, not just fitting the training folds better."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 3) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        self._model = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingReturnModel:
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class ARIMAReturnModel:
    """ARIMA(p, d, q) fit on the return series itself, ignoring the lagged
    feature matrix (ARIMA already models its own lag structure). Kept
    behind the same fit(X, y)/predict(X) interface as the others so it
    slots into walk_forward_validate unchanged - X is only used for its
    length, to know how many steps ahead to forecast."""

    def __init__(self, order: tuple[int, int, int] = (1, 0, 0)) -> None:
        self.order = order
        self._fitted = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> ARIMAReturnModel:
        from statsmodels.tsa.arima.model import ARIMA

        self._fitted = ARIMA(y, order=self.order).fit()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        steps = len(X)
        return np.asarray(self._fitted.forecast(steps=steps))


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    rmse: float
    mae: float
    squared_errors: np.ndarray  # per-test-point (pred - actual)**2, for paired comparisons


def walk_forward_validate(
    model_factory: Callable[[], Model],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> list[FoldResult]:
    """Expanding-window walk-forward validation: each fold trains on
    everything before it and tests on the later, unseen block."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X_arr, y_arr = X.to_numpy(), y.to_numpy()
    results = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X_arr), start=1):
        model = model_factory()
        model.fit(X_arr[train_idx], y_arr[train_idx])
        preds = model.predict(X_arr[test_idx])
        squared_errors = (preds - y_arr[test_idx]) ** 2
        rmse = float(np.sqrt(squared_errors.mean()))
        mae = float(mean_absolute_error(y_arr[test_idx], preds))
        results.append(
            FoldResult(
                fold=i, n_train=len(train_idx), n_test=len(test_idx), rmse=rmse, mae=mae,
                squared_errors=squared_errors,
            )
        )
    return results
