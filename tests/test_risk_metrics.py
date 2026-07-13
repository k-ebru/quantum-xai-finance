import numpy as np
import pandas as pd

from src.risk_metrics import (
    historical_cvar,
    historical_var,
    monte_carlo_cvar,
    monte_carlo_var,
    stress_test,
)


def test_historical_tail_metrics():
    returns = pd.Series([0.01, -0.02, 0.03, -0.04, 0.0])

    assert np.isclose(historical_var(returns, 0.95), 0.036)
    assert np.isclose(historical_cvar(returns, 0.95), 0.04)


def test_monte_carlo_metrics_are_reproducible():
    returns = pd.Series([0.01, -0.02, 0.03, -0.04, 0.0])

    assert monte_carlo_var(returns) == monte_carlo_var(returns)
    assert monte_carlo_cvar(returns) == monte_carlo_cvar(returns)


def test_stress_window_reports_observation_count():
    dates = pd.date_range("2020-01-01", periods=10)
    returns = pd.Series(np.linspace(-0.05, 0.02, 10), index=dates)

    result = stress_test(returns, "2020-01-03", "2020-01-06")

    assert result["n_observations"] == 4
