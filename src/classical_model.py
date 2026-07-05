"""Baseline classical model and SHAP explanations for the stress regime task."""

import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

RANDOM_STATE = 42


def train_baseline_model(X_train, y_train) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "auc": roc_auc_score(y_test, probs),
        "brier": brier_score_loss(y_test, probs),
    }


def compute_shap_values(model, X):
    explainer = shap.Explainer(model, X)
    return explainer(X)
