---
name: train-imbalanced-pipeline
description: Train a classification pipeline on an imbalanced dataset following an evidence-based decision framework (not the default "just resample it" instinct) — choosing the right metric based on whether calibrated probabilities, pure discrimination, or a hard classification decision is needed, deriving or tuning a decision threshold instead of defaulting to 0.5, and treating resampling/SMOTE/class weights as a last resort rather than a first move. Use this whenever the user wants to train, build, or improve a model on imbalanced/rare-event data, mentions class imbalance, minority class detection, fraud/churn/disease/default prediction, or asks about SMOTE, undersampling, oversampling, class weights, or probability calibration for such a dataset — even if they don't name this skill directly. Always clarifies the modeling objective (calibration vs discrimination vs decision) and any known misclassification costs before writing training code, asking only for what the user hasn't already stated.
---

# Train an imbalanced-data pipeline

Trains a classifier on imbalanced/rare-event data the way the evidence
actually supports, not the way folk wisdom does. The core finding this
skill is built around: **class imbalance itself rarely breaks model
performance — poor methodology does.** Reflexively reaching for SMOTE,
undersampling, or `class_weight="balanced"` is always the wrong
first move; choosing the right model, the right metric for the actual goal,
and the right decision threshold gets you further, preserves calibration,
and doesn't throw away or fabricate data.

## Before writing any training code: get the decisions right

Read `references/decision_framework.md` and work through it *before*
touching the model. It lays out the questions that determine everything
downstream — principally: **does the model's output need to be a calibrated
probability, or just rank cases well, or become a hard label?** and **do
real misclassification costs exist, or does the threshold need to be tuned
empirically?**

The user may have already answered some or all of this in their request
("we need calibrated fraud probabilities for underwriters," "just flag the
riskiest 5%," "false negatives cost us about $200 each, false positives
about $5"). Extract whatever they've already told you and don't ask about
it again. For whatever's genuinely unstated, ask directly — a couple of
short, concrete questions beats guessing and building the wrong pipeline.
Don't ask about constraints that plainly don't apply (e.g. don't ask about
computational limits on a 10,000-row dataset).

If, after asking, the user still doesn't know or care, `decision_framework.md`
has sane defaults (discrimination as the objective, a couple of strong
ensemble models plus logistic regression, no resampling) — use them rather
than blocking.

## Then: build the pipeline

1. **Look at the data first.** Class counts, minority class size, whether
   features are continuous or mixed, whether there's an existing
   preprocessing pipeline (or an EDA notebook, e.g. from the `eda-notebook`
   skill) to build on. If the minority class is very small, mention that a
   learning curve (performance vs training set size) is worth checking
   before investing further — this is a data-sufficiency question no
   modelling choice fixes.

2. **Pick model candidates.** Default to a couple of strong ensemble models
   (random forest, and a gradient boosting implementation actually
   available in the environment — XGBoost/LightGBM/CatBoost if installed,
   else `sklearn.ensemble.GradientBoostingClassifier` or
   `HistGradientBoostingClassifier`) plus logistic regression as a fast,
   often-calibrated baseline. Only narrow this down if the user has stated
   a real constraint (interpretability requirement, deployment environment,
   etc.) — don't default to a single model "because it's usually best,"
   comparing a couple of options is cheap and avoids that trap.

3. **Train on the original class distribution.** No resampling, no class
   weights, by default. This isn't a placeholder — see
   `references/cost_sensitive_and_resampling.md` for why, and for the
   narrow, explicit conditions under which resampling is actually justified
   (check them before reaching for `imbalanced-learn`, don't reach for it
   out of habit).

4. **Optimise and evaluate with the metric the decisions call for**, per
   `references/metrics_and_calibration.md`:
   - Calibrated probabilities → log loss / Brier score, verified with a
     calibration curve on held-out data, recalibrated (Platt or isotonic)
     if needed.
   - Discrimination only → ROC-AUC or average precision.
   - Hard classification decision → the chosen classification metric, always
     at a *tuned* threshold (see step 5) — never sklearn's default 0.5.

   Always report **uncertainty** (bootstrap or cross-validation mean ± std),
   never a single point value — use `scripts/imbalanced_eval.py`'s
   `bootstrap_evaluate` rather than a one-shot `.score()` call. Small
   differences between models are frequently within noise; say so when it's
   true instead of declaring a winner from one number.

5. **If a hard decision/threshold is needed**, follow
   `references/cost_sensitive_and_resampling.md` Step 1: derive it from a
   real cost matrix if one exists (`cost_matrix_to_threshold`), or search
   for it empirically on validation folds (`optimal_threshold` /
   `bootstrap_optimal_threshold` / `TunedThresholdClassifierCV`). Do this
   search on validation data or CV folds, never on the final test set.

6. **Always use stratified splits/CV** (`train_test_split(..., stratify=y)`,
   `StratifiedKFold`) — with a rare minority class, an unstratified split
   can easily leave a fold with too few (or zero) positive examples.

7. **Verify before calling it done**: check that discrimination and
   calibration (if relevant) were actually evaluated on data the model/
   calibrator never saw during fitting, that the reported metric came from
   an optimised (not default) threshold if the task called for one, and
   that if resampling was used, it's confined to training folds only inside
   a leakage-safe `imblearn.pipeline.Pipeline` — not a plain scikit-learn
   `Pipeline`.

## Reference files

- `references/decision_framework.md` — the questions to resolve before
  building anything, and why each one matters.
- `references/metrics_and_calibration.md` — metric selection, calibration
  diagnosis and recalibration (Platt vs isotonic), and model-specific
  calibration behaviour.
- `references/cost_sensitive_and_resampling.md` — turning costs into a
  threshold, the resampling/thresholding/class-weighting equivalence, and
  exactly when (rarely) resampling is defensible.
- `scripts/imbalanced_eval.py` — bootstrap evaluation, threshold search
  (single-point and bootstrap), cost-to-threshold, and resampling-ratio
  utilities. Import these rather than re-deriving the same loops inline —
  the same bootstrap-with-uncertainty pattern applies everywhere in this
  skill.

## What NOT to do

- Don't default to accuracy, or to any classification metric at the
  default 0.5 threshold, on imbalanced data.
- Don't calculate threshold-dependent metrics unless the user 
  specifically asks for one or more of them.
- Don't apply resampling or class weights and call it done without
  comparing against a threshold-tuned baseline on the original
  distribution — the comparison is the point.
- Don't claim a model is calibrated because it has good ROC-AUC — check
  with an actual calibration curve on held-out data.
- Don't conclude about calibration without looking at the number of 
  observations per bin. If there are few observations, that point is 
  unreliable and we can't conclude anything.
- Don't fit and evaluate a calibrator (or a threshold) on the same data
  used to fit the base model.
- Don't pick PR-AUC over ROC-AUC (or vice versa) "because the data is
  imbalanced" — that's not a real distinction; pick based on whether a
  holistic (ROC) or minority-focused (PR) view is wanted.
- Don't extrapolate a resampling benefit seen on a weak classifier (KNN,
  single decision tree) to the strong ensemble model actually being used.
