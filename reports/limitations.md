# Limitations

## Quantum model

- The VQC runs on PennyLane's `default.qubit` state vector simulator, not real quantum hardware.
  No noise model, no decoherence, no hardware connectivity constraints.
- 4 qubits, one per feature. This is a small feature budget: the classifier only sees
  `portfolio_vol`, `vix`, `momentum` and `yield_spread`, ranked by SHAP importance on the
  classical model. Adding more features would mean more qubits, which slows the simulator down
  fast on a laptop.
- The circuit uses `BasicEntanglerLayers` with 2 layers, a generic hardware-efficient ansatz. It
  was not chosen or tuned specifically for this classification task.
- Training uses plain gradient descent for 60 steps. The simulator has no batching, so each step
  evaluates the circuit once per training sample. On the full 1588-row training set this measured
  at roughly 24 seconds per step (about 24 minutes for 60 steps), so notebooks 04 and 06 both
  train on a stratified 300-row subsample instead. This is a practical constraint of the
  simulator, documented in the notebooks rather than worked around silently.
- Given all of the above, the quantum classifier (AUC 0.962) does not beat the classical gradient
  boosting baseline (AUC 0.988). This is the expected outcome for this setup, not a surprising
  negative result, and it is reported as such rather than tuned to look better.

## Quantum explainability

- The gradient x input attribution method used in notebook 06 is not a standard, peer-reviewed
  quantum explainability technique. It is an exploratory adaptation of the classical
  gradient-times-input idea, using PennyLane's parameter-shift gradient of the circuit output
  with respect to the input features.
- Agreement with SHAP was partial (Spearman rank correlation 0.800 on 4 features, not
  statistically significant at that sample size).
- The `momentum` feature's attribution is consistently near zero in the quantum method across
  all tested samples and across 20 independent random weight sets, confirming this is a
  structural property of the circuit rather than an artifact of the trained weights. `momentum`
  maps to wire 2, which sits diametrically opposite the measured wire (wire 0) in the 4-qubit
  CNOT ring, and the gradient with respect to that wire's input angle appears to cancel
  structurally. See notebook 06 for the verification.

## Dataset

- The dataset covers roughly 2018 to 2026-07-10, about 2100 trading days. This is a small
  sample by financial machine learning standards, particularly for the stress class (about 530
  labeled stress days out of 2118 after feature lagging).
- The stress label is a simplification: realized portfolio volatility above the 75th percentile
  over the whole sample, not a rolling or regime-switching definition. A fixed whole-sample
  threshold means the label implicitly uses some information from the full history, though the
  features used to predict it are still properly lagged one day.
- The rolling volatility window used for both the label and the key feature `portfolio_vol` is
  20 days. With a one-day lag on the feature, the feature window and the label window share 19
  out of 20 days of return data. The task is therefore close to predicting whether a persistent
  volatility regime continues into the next day rather than forecasting a fresh regime from
  independent information. The persistence baseline (yesterday's label as today's prediction)
  achieves 0.985 AUC on the test set, and the gradient boosting model adds only marginal lift
  over that. Results should be read in that context.
- Class imbalance is moderate (about 3:1 normal to stress days in train, about 11:1 in test due
  to a calmer market period) and was not addressed with resampling or class weighting in either
  model.

## ESG data

- No ESG (environmental, social, governance) data is included. yfinance's ESG data coverage is
  inconsistent and not reliable enough to use as a project input, so it was left out rather than
  included with a caveat.

## Risk metrics

- Monte Carlo VaR/CVaR assumes normally distributed returns (`numpy`'s `rng.normal` sampling from
  the sample mean and standard deviation). The historical VaR/CVaR numbers in this project are
  consistently higher at the 99% confidence level, which points to fat tails that the normal
  assumption does not capture. This gap is expected and is discussed in notebook 05, not treated
  as an error in either method.
- The stress test windows (COVID crash, 2022 rate hike) are fixed calendar dates chosen because
  they are well known stress periods, not detected algorithmically from the data.
