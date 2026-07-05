# Explainable AI for Quantum-Enhanced Financial Risk Models

Small-scale research project comparing a classical gradient boosting classifier against a
variational quantum classifier (VQC) on the task of predicting market stress regimes, with SHAP
and gradient-based explainability applied to both. Also includes historical and Monte Carlo
VaR/CVaR calculations with stress-window comparisons.

Data is real market data pulled from Yahoo Finance (sector ETFs, VIX, treasury yields), not
synthetic. The goal is a working, honestly reported project, not a maximally impressive one: the
quantum model does not beat the classical baseline here, and that result is reported and
explained rather than hidden.

## What's in here

- **Feature engineering**: portfolio volatility, VIX, yield spread, momentum, all lagged one day
  relative to a binary stress label (top 25% realized volatility days)
- **Classical baseline**: gradient boosting classifier with SHAP explainability
- **Quantum classifier**: 4-qubit angle-encoded VQC (PennyLane), trained on the same 4
  SHAP-ranked features
- **Quantum explainability experiment**: gradient x input attribution compared against SHAP on a
  handful of test samples
- **Risk metrics**: historical and Monte Carlo VaR/CVaR, with two real stress windows (the March
  2020 selloff and the 2022 rate hike period)

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate    # on Windows; use .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

## Running

Notebooks are meant to be run in order, 01 through 06, since each one reads the output of the
previous one from `data/`.

```bash
cd notebooks
jupyter notebook
```

Or from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_collection.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_classical_model_shap.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_quantum_classifier.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_var_cvar_stress_test.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_quantum_explainability_experiment.ipynb
```

Notebook 04 and 06 both train the VQC (~3-4 minutes each on a laptop CPU) since the state vector
simulator evaluates the circuit once per sample per gradient step with no batching.

## Results

### Classification: stress regime prediction

| Model | AUC | Brier score |
|---|---:|---:|
| Classical (gradient boosting) | 0.988 | 0.015 |
| Quantum (VQC, 4 qubits) | 0.962 | 0.082 |

The quantum model does not beat the classical baseline. See notebook 04 for a discussion of why
(qubit count, ansatz depth, training budget).

### Feature importance (SHAP, classical model)

`portfolio_vol` dominates by a wide margin, consistent with volatility clustering: yesterday's
realized volatility is the strongest available predictor of today's regime. `vix` and `momentum`
contribute secondary signal, `yield_spread` the least.

### Quantum vs classical explainability

Comparing SHAP against quantum gradient x input attribution on 5 test samples: both methods rank
`portfolio_vol` first and `momentum` last, but disagree on the middle ranking (`vix` vs
`yield_spread`). Spearman rank correlation is 0.800, not statistically significant with only 4
features. See notebook 06 for a specific, unexplained finding: the quantum method's `momentum`
attribution is consistently near zero (~1e-16) across all sampled days.

### VaR / CVaR

| Confidence | Method | VaR | CVaR |
|---|---|---:|---:|
| 95% | Historical | 1.57% | 2.72% |
| 95% | Monte Carlo | 1.84% | 2.32% |
| 99% | Historical | 3.08% | 4.96% |
| 99% | Monte Carlo | 2.63% | 3.03% |

Historical 99% VaR exceeds the Monte Carlo estimate, consistent with fat tails in the actual
return distribution that a normal-distribution Monte Carlo assumption underestimates.

### Stress windows

| Period | 95% VaR | 95% CVaR | Days |
|---|---:|---:|---:|
| Full sample | 1.57% | 2.72% | 2139 |
| COVID crash (Feb-Apr 2020) | 8.62% | 10.11% | 41 |
| 2022 rate hike | 2.19% | 2.91% | 251 |

COVID-period VaR is 5.5x the full-sample VaR; the 2022 rate hike period is 1.4x.

## Project structure

```
quantum-xai-finance/
  README.md
  requirements.txt
  src/
    data_fetch.py             fetch sector ETFs and risk factors from Yahoo Finance
    features.py                feature engineering, stress labeling
    classical_model.py         gradient boosting baseline, SHAP
    quantum_model.py           VQC (PennyLane), quantum gradient attribution
    risk_metrics.py            VaR, CVaR, stress testing
  notebooks/
    01_data_collection.ipynb
    02_feature_engineering.ipynb
    03_classical_model_shap.ipynb
    04_quantum_classifier.ipynb
    05_var_cvar_stress_test.ipynb
    06_quantum_explainability_experiment.ipynb
  data/
    raw/                      downloaded prices (not tracked in git)
    processed/                engineered features, saved model results
  figures/                    all plots generated by the notebooks
  reports/
    limitations.md
```

## Limitations

See [`reports/limitations.md`](reports/limitations.md) for the full list. In short: the quantum
model runs on a laptop-scale simulator with 4 qubits, the feature count is capped by the qubit
budget, the quantum explainability method is exploratory rather than a validated technique, and
the dataset is a few thousand trading days, not a large-scale financial dataset.

## Stack

Python, pandas, NumPy, scikit-learn, statsmodels, SHAP, PennyLane, yfinance, Matplotlib, SciPy.

## License

MIT License.
