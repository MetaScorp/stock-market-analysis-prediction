import numpy as np
import pandas as pd

from stockanalysis.regime.markov_switching import fit_markov_switching


def _two_regime_returns(seed: int = 0, n: int = 1200, block: int = 100):
    rng = np.random.default_rng(seed)
    regimes = (np.arange(n) // block) % 2
    sigmas = np.where(regimes == 0, 0.005, 0.03)
    returns = pd.Series(rng.normal(0.0, sigmas))
    return returns, regimes


def test_fit_converges_on_a_clear_two_regime_series():
    returns, _ = _two_regime_returns()
    result = fit_markov_switching(returns)
    assert result.converged is True


def test_recovers_distinct_low_and_high_vol_regimes():
    returns, _ = _two_regime_returns()
    result = fit_markov_switching(returns)
    low, high = sorted(result.regime_std)
    assert low < 0.01
    assert high > 0.02


def test_transition_matrix_rows_sum_to_one():
    returns, _ = _two_regime_returns()
    result = fit_markov_switching(returns)
    row_sums = result.transition_matrix.sum(axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0], atol=1e-6)


def test_expected_duration_is_positive_and_close_to_true_block_length():
    returns, _ = _two_regime_returns(block=100)
    result = fit_markov_switching(returns)
    durations = result.expected_duration()
    assert (durations > 0).all()
    # true regimes last ~100 periods; the fit shouldn't be wildly off
    assert all(30 < d < 400 for d in durations)


def test_regime_series_is_aligned_with_input_index():
    returns, _ = _two_regime_returns()
    result = fit_markov_switching(returns)
    assert list(result.regime.index) == list(returns.index)


def test_summary_reports_convergence():
    returns, _ = _two_regime_returns()
    text = fit_markov_switching(returns).summary()
    assert "Converged: True" in text
