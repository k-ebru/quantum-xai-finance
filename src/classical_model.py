"""Baseline classical model and SHAP explanations for the stress regime task."""

import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

RANDOM_STATE = 42


def train_baseline_model(X_train, y_train) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_volatility_baseline(X_train, y_train) -> LogisticRegression:
    """Fit a one-feature baseline using yesterday's portfolio volatility."""
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train[["portfolio_vol"]], y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs),
        "brier": brier_score_loss(y_test, probs),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_shap_values(model, background, X):
    explainer = shap.Explainer(model, background)
    return explainer(X)
