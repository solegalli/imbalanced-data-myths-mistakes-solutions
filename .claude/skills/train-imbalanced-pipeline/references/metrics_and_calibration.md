# Metrics and calibration

This assumes `decision_framework.md` has already established what the
model's output is for. This file is about implementing that choice
correctly.

## Picking the metric

| Objective (from decision_framework.md)         | Optimise / model-select with              | Evaluate with |
|--------------------------------------------------|--------------------------------------------|----------------|
| Calibrated probabilities                          | log loss or Brier score                    | log loss / Brier score **and** a calibration curve — the scoring rules conflate calibration with discrimination, so a lower score doesn't by itself prove better calibration |
| Discrimination / ranking only                     | ROC-AUC or average precision (PR-AUC)      | same, plus bootstrap or CV std, never a single point estimate |
| Hard classification decision                      | whatever classification metric matches the business objective (balanced accuracy, F1, recall-at-precision, MCC, or a custom cost/profit function) | same metric, always at its *optimised* threshold — never the sklearn default of 0.5. See `cost_sensitive_and_resampling.md` for how to find that threshold. |

Never use plain accuracy on imbalanced data — a model that always predicts
the majority class scores near the imbalance ratio while catching zero
minority cases. If a class-balanced view of accuracy is wanted, use balanced
accuracy (mean of per-class recall) instead, and be aware it can hide *which*
class is underperforming — check per-class precision/recall alongside it.

**ROC-AUC vs PR-AUC**: there is no evidence that PR curves are inherently
better than ROC curves for imbalanced data — this is a persistent but
unsupported claim. If one model's curve dominates in ROC space it typically
dominates in PR space too. Choose based on what you actually want to see:
ROC-AUC uses all four confusion-matrix quadrants (a holistic view); PR-AUC
only reflects the minority class's precision/recall trade-off (a narrower,
minority-focused view). Don't pick PR-AUC "because the data is imbalanced."

**Whenever a metric requires converting probabilities to labels** (accuracy,
precision, recall, F1, balanced accuracy, MCC), remember scikit-learn's
`.predict()` silently applies a 0.5 cutoff. Use `.predict_proba()` and apply
your own threshold — see `cost_sensitive_and_resampling.md`.

## Model choice and calibration behaviour

Not every model produces trustworthy probabilities by default, regardless of
how well it discriminates:

- **Logistic regression** and **modern gradient boosting** (XGBoost,
  LightGBM, CatBoost) optimise log loss directly, so they tend to be
  reasonably well calibrated out of the box. Good defaults when calibration
  matters.
- **Random forests** are inconsistent — sometimes well calibrated, sometimes
  not. Always check, don't assume.
- **AdaBoost / max-margin methods (SVMs, boosted stumps)** tend to push
  predictions toward the middle of the range (underconfident) — decent
  discrimination, unreliable probabilities.
- **Naive Bayes** tends to push predictions toward 0/1 (overconfident) —
  often needs recalibration if probabilities matter.

If calibration is required, prefer models from the first group and still
verify empirically — never assume calibration, check it.

## Verifying calibration: reliability diagrams

Use `sklearn.calibration.CalibrationDisplay` (or build the curve manually)
on a held-out set, not the training set. Always construct a bar plot with 
the number of observations per probability bin to explore areas with low 
representation. Two things distort the read:

- **Too small a test set** or **too few bins**: unreliable bin averages.
  **Too many bins**: noisy curve. There's no universal right number —
  try a few bin counts and prefer equal-frequency bins over equal-width
  bins if predictions are concentrated near one end of [0, 1] (which they
  will be under heavy imbalance).
- **Class imbalance itself**: predictions concentrate near 0, leaving the
  upper-probability region thin on data and the curve unreliable there,
  regardless of true calibration. A bigger held-out set helps but doesn't
  fully solve this under severe imbalance — say so rather than
  overinterpreting a noisy upper-range curve.

## Recalibration (if the model isn't calibrated and needs to be)

Use `sklearn.calibration.CalibratedClassifierCV`, with one of two setups
depending on how much data is available:

- **Enough data for a 3-way split** (train / calibration / test): fit the
  base model on the training set, wrap it with
  `sklearn.frozen.FrozenEstimator` (sklearn >= 1.6; on older sklearn, refit
  is unavoidable — check the installed version first) so `CalibratedClassifierCV`
  only fits the calibrator, not the base model, and fit that on the
  calibration set. Evaluate on the untouched test set.
- **Not enough data for a 3-way split**: use
  `CalibratedClassifierCV(base_model, method=..., cv=5)` directly on the
  training set — it internally fits the base model on 4 folds and
  calibrates on the held-out fold, repeated 5 times, avoiding a dedicated
  calibration split at the cost of some extra compute.

Either way: **never evaluate the calibrator on data used to fit it or the
base model** — that's the single most common mistake here and it produces
overly optimistic calibration curves.

Choosing between the two calibration methods:
- **Platt scaling** (`method="sigmoid"`): fits a sigmoid; more constrained,
  generalises better with a small calibration set, best when the
  miscalibration itself looks sigmoid-shaped.
- **Isotonic regression** (`method="isotonic"`): fits a flexible monotonic
  step function; needs more data, more prone to overfitting on small or
  highly imbalanced calibration sets, but can correct non-sigmoid
  distortions Platt scaling can't.

Default to Platt scaling when the calibration set is small or the data is
heavily imbalanced (positive examples are scarce in the calibration set);
prefer isotonic when there's ample, only moderately imbalanced data.

## The single most important gotcha

**Class weights and resampling both distort calibration.** They change the
effective class prior the model is trained on, which pushes predicted
probabilities away from the true frequencies even when discrimination is
unaffected. If calibration matters at all, this is a hard constraint: don't
apply class weights, don't resample, full stop — train on the original
distribution. This is one of the reasons `decision_framework.md` treats
resampling as opt-in rather than default, and it's a non-negotiable
consequence of question 1 in that framework being answered "calibrated
probabilities."
