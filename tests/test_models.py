import numpy as np
import pandas as pd

from src.classical_model import (
    evaluate_model,
    train_baseline_model,
    train_volatility_baseline,
)


def _sample_data():
    rng = np.random.default_rng(42)
    portfolio_vol = rng.uniform(0.05, 0.35, 240)
    X = pd.DataFrame(
        {
            "portfolio_vol": portfolio_vol,
            "vix": portfolio_vol * 80 + rng.normal(0, 2, 240),
            "yield_spread": rng.normal(0.5, 0.4, 240),
            "momentum": rng.normal(0, 0.03, 240),
        }
    )
    y = pd.Series((portfolio_vol + rng.normal(0, 0.03, 240) > 0.24).astype(int))
    return X.iloc[:180], X.iloc[180:], y.iloc[:180], y.iloc[180:]


def test_model_evaluation_includes_threshold_metrics():
    X_train, X_test, y_train, y_test = _sample_data()
    model = train_baseline_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    expected = {
        "auc", "pr_auc", "brier", "precision", "recall",
        "tn", "fp", "fn", "tp",
    }
    assert set(metrics) == expected
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == len(y_test)


def test_volatility_baseline_uses_one_column():
    X_train, X_test, y_train, y_test = _sample_data()
    model = train_volatility_baseline(X_train, y_train)
    metrics = evaluate_model(model, X_test[["portfolio_vol"]], y_test)

    assert 0 <= metrics["auc"] <= 1
    assert model.n_features_in_ == 1
