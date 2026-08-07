"""
Reusable evaluation utilities for classifiers trained on imbalanced data.

Why this exists: a single point estimate of a metric on one test set doesn't
tell you whether a difference between two models is real or just sampling
noise, and classification metrics computed at the default 0.5 threshold
rarely reflect what a model can actually do. These functions always report
uncertainty (bootstrap mean +/- std) and, for threshold-dependent metrics,
always search for the operating point instead of assuming 0.5.

Import this rather than re-deriving the same loop in every notebook/script.
"""
from __future__ import annotations

import numpy as np


def bootstrap_evaluate(model, X, y, scorers, n_boot=5, sample_frac=0.6, seed0=1):
    """Evaluate a fitted model on bootstrap samples of (X, y).

    `scorers` is a dict of {name: callable(y_true, y_score_or_proba) -> float}.
    The model must already be fitted; this function only resamples the
    evaluation data, it does not refit the model.

    Returns a dict {name: (mean, std)}.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = int(sample_frac * len(X))

    results = {name: [] for name in scorers}

    for seed in range(seed0, seed0 + n_boot):
        idx = np.random.default_rng(seed).choice(len(X), size=n, replace=True)
        xs, ys = X[idx], y[idx]
        proba = model.predict_proba(xs)[:, 1]
        for name, fn in scorers.items():
            results[name].append(fn(ys, proba))

    return {name: (float(np.mean(v)), float(np.std(v))) for name, v in results.items()}


def optimal_threshold(y_true, proba, metric_fn, thresholds=None):
    """Find the decision threshold that maximises `metric_fn(y_true, y_pred)`.

    `metric_fn` takes (y_true, y_pred_binary) -> float, e.g.
    sklearn.metrics.balanced_accuracy_score or a custom cost/profit function.
    Defaults to searching every unique predicted probability, which is exact
    but can be slow on large datasets -- pass an explicit `thresholds` array
    (e.g. np.linspace(0.01, 0.99, 100)) to speed it up.
    """
    proba = np.asarray(proba)
    if thresholds is None:
        thresholds = np.unique(proba)
    scores = [metric_fn(y_true, proba >= t) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def bootstrap_optimal_threshold(y_true, proba, metric_fn, n_boot=5, sample_frac=1.0, seed0=1, thresholds=None):
    """Like `optimal_threshold`, but repeated over bootstrap samples so you
    get a mean +/- std for both the threshold and the metric it achieves.

    Use this instead of a single-point threshold search whenever the chosen
    threshold will be reused elsewhere (e.g. in production) -- it tells you
    how much the "optimal" cut-off would move under a different sample.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = int(sample_frac * len(y_true))

    best_thresholds, best_scores = [], []
    for seed in range(seed0, seed0 + n_boot):
        idx = np.random.default_rng(seed).choice(len(y_true), size=n, replace=True)
        t, s = optimal_threshold(y_true[idx], proba[idx], metric_fn, thresholds=thresholds)
        best_thresholds.append(t)
        best_scores.append(s)

    return {
        "threshold_mean": float(np.mean(best_thresholds)),
        "threshold_std": float(np.std(best_thresholds)),
        "score_mean": float(np.mean(best_scores)),
        "score_std": float(np.std(best_scores)),
    }


def cost_matrix_to_threshold(c10, c01, c00=0.0, c11=0.0):
    """Derive the theoretical cost-optimal probability threshold p*.

    c10 = cost of predicting positive when the true class is negative (FP cost)
    c01 = cost of predicting negative when the true class is positive (FN cost)
    c00, c11 = cost of the two correct predictions (usually 0)

    Assumes the model outputs calibrated probabilities. If it doesn't,
    calibrate it first (see references/metrics_and_calibration.md) or use
    `bootstrap_optimal_threshold` with a cost/profit-based metric_fn instead,
    which doesn't require calibrated probabilities.
    """
    denom = (c10 - c00) + (c01 - c11)
    if denom == 0:
        raise ValueError("c10 - c00 + c01 - c11 must be non-zero to derive a threshold")
    return (c10 - c00) / denom


def resampling_ratio_from_threshold(p_star, p0=0.5):
    """Multiplier for the number of negative-class examples that would make
    a classifier trained on the resampled data and evaluated at threshold
    p0 (typically 0.5) make the same decisions as one trained on the
    original distribution and evaluated at threshold p_star.

    This is provided for completeness / diagnostic purposes (e.g. to show a
    stakeholder that resampling and threshold-tuning are two paths to the
    same decision rule) -- prefer threshold tuning over actually resampling.
    """
    return (p_star / (1 - p_star)) * ((1 - p0) / p0)
