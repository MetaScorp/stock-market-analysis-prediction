"""Project-wide default settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    tickers: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "AMZN")
    start_date: str = "2018-01-01"
    end_date: str | None = None  # None = today

    trading_days_per_year: int = 252

    random_seed: int = 42

    # Monte Carlo defaults
    mc_simulations: int = 10_000
    mc_horizon_days: int = 252

    # Risk defaults
    var_confidence: float = 0.95


DEFAULT_SETTINGS = Settings()
