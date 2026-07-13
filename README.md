# Market Stress Modelling with Classical and Quantum Methods

This project compares a one-feature volatility baseline, gradient boosting and a small
variational quantum classifier on next-day market stress labels. It also includes SHAP and
quantum attribution experiments, plus historical and Monte Carlo VaR/CVaR calculations.

The input is daily Yahoo Finance data for eight US sector ETFs, VIX and two Treasury yield
series. The download is pinned through 10 July 2026. Raw prices are downloaded locally;
processed features and result tables are kept in the repository.

## What is included

- An expanding stress threshold based only on volatility observed before each label date
- Four one-day-lagged features: portfolio volatility, VIX, yield spread and momentum
- A logistic baseline using only lagged portfolio volatility
- A gradient boosting model with SHAP explanations
- A four-qubit VQC running on PennyLane's state-vector simulator
- Historical and normally distributed Monte Carlo VaR/CVaR
- Full-sample comparisons with the 2020 COVID selloff and the 2022 rate-hike year

## Setup

Python 3.12 was used for the checked results. Package versions are pinned in
`requirements.txt`.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running

Run the notebooks in order because each one saves input for the next:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_collection.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_classical_model_shap.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_quantum_classifier.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_var_cvar_stress_test.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_quantum_explainability_experiment.ipynb
```

Notebook 04 is the slow step. Its 60-step VQC training took 852 seconds on the machine used for
the checked run. Runtime depends on CPU and PennyLane version. Notebook 06 loads the weights saved
by notebook 04 instead of training the same model again.

## Classification results

The split is chronological. Training covers 1 February 2019 to 26 August 2024. The held-out
period covers 27 August 2024 to 10 July 2026 and contains 28 stress days out of 467 observations.

| Model | AUC | PR-AUC | Brier | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Lagged-volatility logistic | 0.998 | 0.976 | 0.018 | 1.000 | 0.821 |
| Gradient boosting | 0.995 | 0.957 | 0.011 | 0.960 | 0.857 |
| VQC, 4 qubits | 0.926 | 0.856 | 0.048 | 0.000 | 0.000 |

AUC measures ranking across all thresholds. PR-AUC focuses on ranking the less common stress
class. Brier measures probability error. Precision and recall use a fixed 0.5 threshold.

The one-feature model has the strongest ranking metrics. Gradient boosting has the better Brier
score and catches one additional stress day at the fixed threshold. The VQC ranks some stress
days correctly, but all of its scores remain below 0.5, so it predicts no stress days at that
threshold. The threshold was not tuned on the test set.

These results do not show a quantum advantage. They mainly show how much of this label can be
explained by volatility persistence.

## Feature attribution

SHAP uses the training period as its background and explains predictions in the held-out period.
The gradient boosting ranking is:

1. `portfolio_vol`
2. `vix`
3. `yield_spread`
4. `momentum`

Notebook 06 compares SHAP with gradient-times-input attribution from the VQC on five test days.
The rankings have Spearman correlation 0.200 with p=0.800, so the small experiment does not
support a claim that the methods agree. A separate check over 20 random weight sets confirms
that all four encoded inputs can affect the revised paired readout.

## VaR and CVaR

The portfolio is the daily equal-weight average of the eight sector ETF returns.

| Confidence | Method | VaR | CVaR |
|---|---|---:|---:|
| 95% | Historical | 1.57% | 2.72% |
| 95% | Monte Carlo | 1.84% | 2.32% |
| 99% | Historical | 3.08% | 4.96% |
| 99% | Monte Carlo | 2.63% | 3.03% |

The Monte Carlo calculation samples one-day returns from a fitted normal distribution. The larger
historical 99% tail estimates are consistent with the skew and excess kurtosis in the observed
returns.

| Period | 95% VaR | 95% CVaR | Days |
|---|---:|---:|---:|
| Full sample | 1.57% | 2.72% | 2,139 |
| COVID selloff, 15 Feb to 15 Apr 2020 | 8.62% | 10.11% | 41 |
| 2022 calendar year | 2.19% | 2.91% | 251 |

The full sample includes both stress windows. It is a reference period, not a separate normal
period.

## Tests

```bash
python -m pytest -q
```

The tests cover the expanding label boundary, feature table integrity, model metric outputs,
VaR/CVaR calculations and quantum readout sensitivity. GitHub Actions runs the same suite on
Python 3.12.

## Project structure

```text
Market-Stress-Models/
  README.md
  requirements.txt
  src/
  notebooks/
  tests/
  data/
    raw/                  downloaded prices, not tracked
    processed/            features, model metrics and VQC weights
  figures/
  reports/
    limitations.md
```

## Limitations

The main limitations are the heuristic stress definition, the small and shifting stress class,
the simulator-only VQC, the 300-row quantum training sample, the fixed classification threshold
and the exploratory attribution comparison. See [`reports/limitations.md`](reports/limitations.md)
for details.

## Stack

Python, pandas, NumPy, scikit-learn, SHAP, PennyLane, yfinance, Matplotlib and SciPy.

## License

MIT License.
