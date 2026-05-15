"""
Utilities for Sprint 5 Evaluation and Business Value analysis.

The project target uses -1 for phishing and 1 for legitimate URLs. For business
value calculations we treat phishing as the positive class because detecting it
is the operational objective.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


PHISHING_LABEL = -1
LEGITIMATE_LABEL = 1
MODEL_PHISHING_LABELS = (-1, 0)


DEFAULT_BUSINESS_ASSUMPTIONS = {
    "benefit_tp": 100.0,
    "cost_fp": -20.0,
    "cost_fn": -80.0,
    "benefit_tn": 0.0,
    "annual_volume": 100_000,
}


def phishing_probability(model, X):
    """Return P(phishing) regardless of class order in a sklearn classifier."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    phishing_model_label = next((label for label in MODEL_PHISHING_LABELS if label in classes), None)
    if phishing_model_label is None:
        raise ValueError(f"Could not identify phishing class in model.classes_: {classes}")
    phishing_idx = classes.index(phishing_model_label)
    return proba[:, phishing_idx]


def labels_from_threshold(y_proba_phishing, threshold=0.5):
    """Map phishing probabilities to project labels using a decision threshold."""
    return np.where(np.asarray(y_proba_phishing) >= threshold, PHISHING_LABEL, LEGITIMATE_LABEL)


def business_value_from_counts(tn, fp, fn, tp, assumptions=None):
    """Calculate total business value from a binary confusion matrix."""
    a = {**DEFAULT_BUSINESS_ASSUMPTIONS, **(assumptions or {})}
    return (
        tp * a["benefit_tp"]
        + fp * a["cost_fp"]
        + fn * a["cost_fn"]
        + tn * a["benefit_tn"]
    )


def evaluate_threshold(y_true, y_proba_phishing, threshold=0.5, assumptions=None):
    """
    Evaluate a threshold in technical and business terms.

    Confusion matrix orientation:
    - positive class: phishing (-1)
    - negative class: legitimate (1)
    """
    y_pred = labels_from_threshold(y_proba_phishing, threshold)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[LEGITIMATE_LABEL, PHISHING_LABEL],
    ).ravel()
    value = business_value_from_counts(tn, fp, fn, tp, assumptions)
    per_case = value / len(y_true)
    annual_volume = (assumptions or DEFAULT_BUSINESS_ASSUMPTIONS).get(
        "annual_volume", DEFAULT_BUSINESS_ASSUMPTIONS["annual_volume"]
    )
    annual_value = per_case * annual_volume
    return {
        "threshold": round(float(threshold), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "value_test": round(float(value), 2),
        "value_per_case": round(float(per_case), 4),
        "annual_value": round(float(annual_value), 2),
    }


def threshold_search(y_true, y_proba_phishing, assumptions=None, start=0.05, stop=0.95, step=0.01):
    """Search the threshold that maximizes expected business value."""
    thresholds = np.arange(start, stop + step, step)
    rows = [
        evaluate_threshold(y_true, y_proba_phishing, threshold=t, assumptions=assumptions)
        for t in thresholds
    ]
    df = pd.DataFrame(rows)
    best = df.loc[df["value_test"].idxmax()].to_dict()
    return df, best


def gain_curve(y_true, y_proba_phishing):
    """Build a cumulative gain curve for the phishing class."""
    y_true = np.asarray(y_true)
    y_proba_phishing = np.asarray(y_proba_phishing)
    positives = (y_true == PHISHING_LABEL).astype(int)
    order = np.argsort(-y_proba_phishing)
    sorted_pos = positives[order]
    total_pos = sorted_pos.sum()

    contacted_share = np.arange(1, len(sorted_pos) + 1) / len(sorted_pos)
    captured_share = np.cumsum(sorted_pos) / total_pos
    lift = captured_share / contacted_share

    return pd.DataFrame({
        "contacted_share": contacted_share,
        "captured_share": captured_share,
        "lift": lift,
    })
