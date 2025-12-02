import matplotlib.pyplot as plt

from stockanalysis.indicators import bollinger_bands
from stockanalysis.risk import monte_carlo_var
from stockanalysis.stats import correlation_matrix, daily_returns
from stockanalysis.viz import (
    plot_correlation_heatmap,
    plot_equity_curve,
    plot_price_with_bands,
    plot_returns_distribution,
    plot_var_distribution,
)


def test_plot_price_with_bands_returns_figure(close):
    bands = bollinger_bands(close)
    fig = plot_price_with_bands(close, bands)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_returns_distribution_returns_figure(close):
    fig = plot_returns_distribution(daily_returns(close))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_correlation_heatmap_returns_figure(ohlcv):
    prices = ohlcv[["Close"]].rename(columns={"Close": "TEST"})
    prices["OTHER"] = prices["TEST"] * 1.01
    fig = plot_correlation_heatmap(correlation_matrix(prices))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_equity_curve_returns_figure(close):
    fig = plot_equity_curve(close, benchmark=close * 0.9)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_var_distribution_returns_figure():
    result = monte_carlo_var(mu=0.0, sigma=0.02, n_simulations=1000, seed=1)
    fig = plot_var_distribution(result.simulated_returns, result.var)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
