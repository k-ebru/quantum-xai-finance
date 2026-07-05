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
  statistically significant at that sample size) and one specific finding, the `momentum`
  feature's attribution being consistently near zero in the quantum method, was reported without
  being fully explained. It looks like a property of this specific trained circuit rather than
  a general result.

## Dataset

- The dataset covers roughly 2018 to the present, a few thousand trading days. This is a small
  sample by financial machine learning standards, particularly for the stress class (about 530
  labeled stress days out of 2118 after feature lagging).
- The stress label is a simplification: realized portfolio volatility above the 75th percentile
  over the whole sample, not a rolling or regime-switching definition. A fixed whole-sample
  threshold means the label implicitly uses some information from the full history, though the
  features used to predict it are still properly lagged one day.
- Class imbalance is moderate (about 3:1 normal to stress days) and was not addressed with
  resampling or class weighting in either model.

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
