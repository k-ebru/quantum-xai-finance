"""Value at Risk and Expected Shortfall (CVaR) calculations."""

import numpy as np
import pandas as pd

RANDOM_STATE = 42


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    return -np.percentile(returns, (1 - confidence) * 100)


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    var = historical_var(returns, confidence)
    tail_losses = returns[returns <= -var]
    return -tail_losses.mean()


def monte_carlo_var(returns: pd.Series, confidence: float = 0.95, n_simulations: int = 10000) -> float:
    rng = np.random.default_rng(RANDOM_STATE)
    mu, sigma = returns.mean(), returns.std()
    simulated = rng.normal(mu, sigma, n_simulations)
    return -np.percentile(simulated, (1 - confidence) * 100)


def monte_carlo_cvar(returns: pd.Series, confidence: float = 0.95, n_simulations: int = 10000) -> float:
    rng = np.random.default_rng(RANDOM_STATE)
    mu, sigma = returns.mean(), returns.std()
    simulated = rng.normal(mu, sigma, n_simulations)
    var = -np.percentile(simulated, (1 - confidence) * 100)
    tail = simulated[simulated <= -var]
    return -tail.mean()


def stress_test(returns: pd.Series, start: str, end: str) -> dict:
    """Recompute historical VaR and CVaR restricted to a known stress window,
    e.g. the March 2020 selloff or the 2022 rate hike period.
    """
    window = returns.loc[start:end]
    return {
        "var_95": historical_var(window),
        "cvar_95": historical_cvar(window),
        "n_observations": len(window),
    }
