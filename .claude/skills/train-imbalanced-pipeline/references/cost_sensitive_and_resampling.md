# Cost-sensitive decisions, thresholds, and (rarely) resampling

The load-bearing fact behind this whole file: **threshold adjustment, class
weighting, and resampling are mathematically equivalent ways of encoding
misclassification costs into a classifier's decisions.** None of them make a
model better at telling the classes apart — they only move where the
decision boundary falls. This is Elkan (2001) and Zadrozny et al. (2003)'s
result, and it holds up empirically on modern ensemble models too. Treat
"resampling/class weights will improve my model" as false by default; the
real question is only ever "what should the decision threshold be."

Use `scripts/imbalanced_eval.py` for all of the threshold-search and
bootstrap-uncertainty code below rather than re-deriving it inline.

## Step 1: get a threshold, not a resampled dataset

**If a real cost matrix exists** (from decision_framework.md question 2):
with `C(1,0)` the cost of a false positive and `C(0,1)` the cost of a false
negative (correct predictions usually cost 0), the theoretical cost-optimal
threshold is:

```
p* = C(1,0) / (C(1,0) + C(0,1))
```

`cost_matrix_to_threshold(c10, c01)` in the bundled script computes this.
It assumes the model's probabilities are calibrated — if they aren't (see
`metrics_and_calibration.md`), either calibrate first or skip straight to
empirical thresholding below, which doesn't need calibrated probabilities.

**If costs are instance-dependent** (e.g. the benefit of a decision scales
with a dollar amount that varies per row, not a fixed class-level cost),
there's no closed-form threshold. Use sample weights equal to the per-row
cost/benefit, or optimise a custom profit function empirically (below).

**If there's no real cost matrix**, empirically search for the threshold
that maximises whatever classification metric matters (from
`metrics_and_calibration.md`):

```python
from imbalanced_eval import optimal_threshold, bootstrap_optimal_threshold

proba = model.predict_proba(X_val)[:, 1]
threshold, score = optimal_threshold(y_val, proba, your_metric_fn)

# Prefer this over the single-point version above -- it also tells you
# how much the threshold and score would move on a different sample:
result = bootstrap_optimal_threshold(y_val, proba, your_metric_fn, n_boot=100)
```

Do threshold search on validation data or cross-validation folds, not the
final test set — a threshold chosen on the test set makes that set no
longer a clean, untouched evaluation of the pipeline.
`sklearn.model_selection.TunedThresholdClassifierCV` does this during
cross-validation automatically; pass a custom grid of thresholds if the
default 100-point grid is too coarse for a highly imbalanced dataset.

## Step 2: when (if ever) resampling actually belongs in the pipeline

Default answer: it doesn't. Train on the original distribution and adjust
the threshold instead — it's simpler, doesn't discard or fabricate data, and
doesn't damage calibration the way resampling does. Empirically, across
modern ensemble models, neither undersampling (random or the neighbourhood-
cleaning variants: ENN, Tomek Links, NCR, OSS, CNN, etc.) nor oversampling
(random or SMOTE and its variants) produces a statistically meaningful
improvement in discrimination (ROC-AUC/PR-AUC) once the threshold is
properly tuned instead of left at 0.5. Most of the historical evidence for
these methods comes from decision trees, KNN, and other weak classifiers
evaluated at the default threshold — it doesn't transfer to gradient
boosting or random forests.

Resampling is defensible only in these narrow cases (all from
`decision_framework.md` question 3):

- **Undersampling as a compute shortcut**: the dataset is so large that
  training time is genuinely prohibitive, and reducing majority-class
  volume is the only practical way to make iteration feasible. This is a
  speed trade-off, not a performance play — say so explicitly if you use it
  this way, and still compare against a threshold-tuned baseline once
  feasible.
- **Oversampling when more data can't be collected and the classifier is
  weak**: if a strong ensemble model genuinely isn't available (e.g. a
  hard interpretability/regulatory constraint forces something like logistic
  regression or a single tree), oversampling has shown modest, real benefit
  for weak learners in the literature. If a strong classifier is on the
  table at all, prefer it over oversampling a weak one.
- **A fixed downstream threshold that can't be changed**: some legacy
  system hard-codes a 0.5 cutoff and there's no way to pass through a
  tuned threshold. Even here, prefer adjusting via sample weights (below)
  over physically resampling the data, since it doesn't touch data
  integrity and is easier to reason about.

If none of these apply, don't reach for `imbalanced-learn`'s samplers —
build the pipeline per Step 1 instead.

## If resampling genuinely is justified: implementation notes

- **Only apply it to training folds, never to validation/test data.** Use
  `imblearn.pipeline.Pipeline` (not scikit-learn's).
- **If you're comparing multiple models or tuning hyperparameters on the
  same resampled data, resample once per fold and reuse it**, rather than
  letting it rerun inside the pipeline for every candidate and every model —
  this matters a lot for the KNN-based cleaning methods (ENN, NCR, CNN,
  etc.), which don't scale well and are wasteful to rerun redundantly.
- **SMOTE (and its variants) assume continuous features** — it generates
  points by interpolating between neighbours, which is meaningless for
  categorical columns. Only use it on datasets that are all (or mostly)
  continuous, or encode/handle categoricals separately first.
- After resampling, still evaluate with a threshold search and bootstrap
  uncertainty (Step 1's tools) — resampled models still need a decision
  threshold, and "resampled + default 0.5" is exactly the flawed setup that
  made resampling look more effective than it is in much of the older
  literature.

## Class weights: same equivalence, easier plumbing

`class_weight="balanced"` (or explicit sample weights) is mathematically the
same move as resampling — it reweights the training objective rather than
physically duplicating/removing rows. It's often the more convenient
implementation (works naturally with cross-validation and hyperparameter
search, no pipeline leakage risk), but it carries the exact same calibration
cost as resampling (see `metrics_and_calibration.md`) and the same "doesn't
actually improve discrimination" result. Use it under the same narrow
conditions as resampling above, not as a casual default, and prefer sample
weights derived from real per-instance costs (see the KDD98 donation-profit
pattern below) over frequency-based weights whenever real costs are known.

## Instance-dependent costs: sample-weight pattern

When cost/benefit varies per row (e.g. a donation amount, a claim value)
rather than being fixed per class, encode it directly as a sample weight and
train normally — no resampling, no fixed threshold derivation needed:

```python
# cost of a false positive is fixed (e.g. mailing cost); benefit of a true
# positive varies per row (e.g. donation amount)
sample_weight = np.where(y_train == 1, benefit_train, fixed_fp_cost)
model.fit(X_train, y_train, sample_weight=sample_weight)
```

Evaluate the resulting model by computing the actual business metric (e.g.
profit) at each candidate threshold and picking the best one empirically —
there's no closed-form threshold when costs vary per instance. If the model
needs to be tuned via cross-validation with this custom objective, scikit-
learn's metadata routing (`sklearn.set_config(enable_metadata_routing=True)`)
lets a custom scorer receive the per-row cost/benefit array alongside
`y_true`/`y_pred` during `TunedThresholdClassifierCV` or `cross_validate`.
