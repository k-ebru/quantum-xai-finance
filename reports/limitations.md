# Limitations

## Dataset and labels

- Prices come from Yahoo Finance and are pinned through 10 July 2026. Historical values can be
  revised upstream, and the raw download is not tracked in Git.
- The feature table starts on 1 February 2019 because the expanding stress threshold requires 252
  prior volatility observations.
- Stress is a heuristic label. A day is positive when its realized portfolio volatility exceeds
  the expanding 75th percentile available before that date. It is not an externally observed
  crisis label or a regime-switching estimate.
- The full feature table has 316 stress days and 1,552 normal days. The rate falls from 20.6% in
  training to 6.0% in the held-out period, so the evaluation includes a meaningful distribution
  shift.
- No resampling or class weighting is used. Precision and recall therefore depend strongly on the
  fixed 0.5 threshold.

## Classical models

- The lagged-volatility logistic model is the main reference. It has higher held-out AUC and
  PR-AUC than gradient boosting, showing that volatility persistence explains most of the
  ranking performance.
- Gradient boosting has a lower Brier score and slightly higher recall at the fixed threshold.
  The comparison is based on one chronological split, not rolling cross-validation.
- SHAP uses the training period as background and the held-out period as the explained sample.
  The feature ranking can still vary across market periods.

## Quantum model

- The VQC runs on PennyLane's `default.qubit` state-vector simulator, not quantum hardware. There
  is no noise model, decoherence or hardware connectivity constraint.
- The circuit uses four qubits, two `BasicEntanglerLayers` and a paired PauliZ readout on wires 0
  and 1. A diagnostic over 20 random weight sets confirms that each input can affect this readout.
- Training uses 60 gradient-descent steps on a reproducible 300-row stratified sample. The full
  1,401-row training set is not used because simulator cost grows quickly.
- Training took 852 seconds on the checked machine. Hardware and library versions affect runtime.
- The VQC has held-out AUC 0.926 and PR-AUC 0.856, but produces no positive predictions at the
  fixed 0.5 threshold. Its probability scale is not ready for operational use.
- No threshold is selected on the test set. A future experiment would need a separate validation
  period for threshold selection or calibration.
- The VQC does not beat either classical model. There is no evidence of quantum advantage in this
  setup.

## Attribution experiment

- Gradient-times-input is an exploratory adaptation, not a validated quantum explainability
  technique.
- SHAP and VQC attributions explain different models after different feature transformations.
  Per-sample normalization makes bar heights comparable but does not make the methods equivalent.
- The comparison uses five held-out days. Spearman correlation is 0.200 with p=0.800, which does
  not support a claim of agreement.
- The circuit-gradient check verifies feature visibility only. It does not validate the
  attribution method or show that the learned feature effects are economically meaningful.

## Risk metrics

- Monte Carlo VaR/CVaR samples one-day returns from a fitted normal distribution. It does not
  model volatility clustering, changing correlations or multi-day paths.
- The 10,000 simulations leave limited observations in the 99% tail. The fixed random seed makes
  the notebook repeatable but does not remove Monte Carlo error.
- The COVID and 2022 windows are fixed calendar choices. They are compared with the full sample,
  which includes those same windows.
- VaR forecasts are not backtested with coverage or independence tests. The notebook is a
  descriptive comparison, not a production risk model.

## Scope

- No ESG data is included. Yahoo Finance coverage was not treated as reliable enough for a
  consistent historical input.
- The project uses a few thousand daily observations and four model features. It is a small
  experiment rather than a large financial forecasting system.
