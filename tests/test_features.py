import numpy as np
import pandas as pd

from src.features import build_feature_matrix, label_stress_regime


def test_stress_label_uses_only_prior_history():
    vol = pd.Series([1.0, 2.0, 3.0, 4.0, 2.0, 5.0])
    labels = label_stress_regime(vol, quantile=0.75, min_history=3)

    assert labels.iloc[:3].isna().all()
    assert labels.iloc[3] == 1

    extended = pd.concat([vol, pd.Series([1000.0])], ignore_index=True)
    extended_labels = label_stress_regime(extended, quantile=0.75, min_history=3)
    pd.testing.assert_series_equal(labels, extended_labels.iloc[: len(labels)])


def test_feature_matrix_is_complete_and_binary():
    dates = pd.bdate_range("2020-01-01", periods=340)
    trend = np.linspace(0, 1, len(dates))
    prices = pd.DataFrame(
        {
            "XLF": 100 + trend * 20 + np.sin(trend * 30),
            "XLK": 90 + trend * 25 + np.cos(trend * 25),
            "^VIX": 15 + np.sin(trend * 18) * 3,
            "^TNX": 2.0 + trend,
            "^IRX": 1.0 + trend * 0.5,
        },
        index=dates,
    )

    features = build_feature_matrix(prices)

    assert not features.empty
    assert not features.isna().any().any()
    assert set(features["stress"].unique()).issubset({0, 1})
    assert features.index.is_monotonic_increasing
