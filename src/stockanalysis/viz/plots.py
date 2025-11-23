"""
Plotting helpers.

Everything returns a Figure instead of calling plt.show(), so it's usable
outside a notebook cell and actually testable without a display backend.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless-safe; callers can still .show() if they want


def plot_price_with_bands(
    prices: pd.Series, bands: pd.DataFrame | None = None, title: str = "Price"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices.index, prices.values, label="Close", color="black", linewidth=1.2)
    if bands is not None:
        ax.plot(bands.index, bands["middle"], label="Middle", linestyle="--")
        ax.plot(bands.index, bands["upper"], label="Upper", alpha=0.6)
        ax.plot(bands.index, bands["lower"], label="Lower", alpha=0.6)
        ax.fill_between(bands.index, bands["lower"], bands["upper"], alpha=0.1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_returns_distribution(returns: pd.Series, bins: int = 50) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(returns.dropna(), bins=bins, color="steelblue", edgecolor="white")
    ax.axvline(returns.mean(), color="black", linestyle="--", label="Mean")
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    ax.set_title("Return Correlation")
    fig.tight_layout()
    return fig


def plot_equity_curve(
    equity: pd.Series, benchmark: pd.Series | None = None, title: str = "Equity Curve"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.values, label="Strategy", linewidth=1.5)
    if benchmark is not None:
        ax.plot(
            benchmark.index, benchmark.values, label="Buy & Hold",
            linestyle="--", alpha=0.8,
        )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_var_distribution(
    simulated_returns: np.ndarray, var: float, confidence: float = 0.95
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(simulated_returns, bins=60, color="slategray", edgecolor="white")
    ax.axvline(-var, color="red", linestyle="--", label=f"VaR ({confidence:.0%})")
    ax.set_title("Monte Carlo Simulated Return Distribution")
    ax.set_xlabel("Simulated Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    return fig
