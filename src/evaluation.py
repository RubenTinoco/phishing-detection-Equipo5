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
    "cost_false_negative": 500.0,
    "cost_false_positive": 25.0,
    "cost_review": 2.0,
    "cost_operational_per_url": 0.50,
    "annual_volume": 100_000,
    "minimum_recall": 0.99,
}


BUSINESS_VALUE_SCENARIOS = {
    "conservador": {
        "cost_false_negative": 250.0,
        "cost_false_positive": 25.0,
        "cost_review": 2.0,
        "cost_operational_per_url": 0.50,
        "annual_volume": 100_000,
        "minimum_recall": 0.98,
    },
    "base": DEFAULT_BUSINESS_ASSUMPTIONS.copy(),
    "severo": {
        "cost_false_negative": 1_000.0,
        "cost_false_positive": 25.0,
        "cost_review": 2.0,
        "cost_operational_per_url": 0.50,
        "annual_volume": 100_000,
        "minimum_recall": 0.995,
    },
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


def business_costs_from_counts(tn, fp, fn, tp, assumptions=None):
    """Calculate cost without model, cost with model, net savings and ROI."""
    a = {**DEFAULT_BUSINESS_ASSUMPTIONS, **(assumptions or {})}
    total_urls = tn + fp + fn + tp
    phishing_real = fn + tp
    alerts = fp + tp

    review_cost = alerts * a["cost_review"]
    operational_cost = total_urls * a["cost_operational_per_url"]
    operational_cost_total = review_cost + operational_cost
    cost_without_model = phishing_real * a["cost_false_negative"]
    cost_with_model = (
        fn * a["cost_false_negative"]
        + fp * a["cost_false_positive"]
        + review_cost
        + operational_cost
    )
    net_savings = cost_without_model - cost_with_model
    roi = np.nan if operational_cost_total == 0 else net_savings / operational_cost_total
    value_per_1000_urls = np.nan if total_urls == 0 else net_savings / total_urls * 1000

    return {
        "total_urls": int(total_urls),
        "phishing_real": int(phishing_real),
        "alerts": int(alerts),
        "cost_without_model": float(cost_without_model),
        "cost_with_model": float(cost_with_model),
        "review_cost": float(review_cost),
        "operational_cost": float(operational_cost),
        "operational_cost_total": float(operational_cost_total),
        "net_savings": float(net_savings),
        "roi": float(roi),
        "value_per_1000_urls": float(value_per_1000_urls),
    }


def business_value_from_counts(tn, fp, fn, tp, assumptions=None):
    """Return net savings from a binary confusion matrix."""
    return business_costs_from_counts(tn, fp, fn, tp, assumptions)["net_savings"]


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
    costs = business_costs_from_counts(tn, fp, fn, tp, assumptions)
    total = len(y_true)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    false_positive_rate = 0.0 if (fp + tn) == 0 else fp / (fp + tn)
    scale = (assumptions or DEFAULT_BUSINESS_ASSUMPTIONS).get(
        "annual_volume", DEFAULT_BUSINESS_ASSUMPTIONS["annual_volume"]
    ) / total
    annual_net_savings = costs["net_savings"] * scale
    annual_cost_without_model = costs["cost_without_model"] * scale
    annual_cost_with_model = costs["cost_with_model"] * scale
    return {
        "threshold": round(float(threshold), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_urls": costs["total_urls"],
        "phishing_real": costs["phishing_real"],
        "alerts": costs["alerts"],
        "recall": round(float(recall), 6),
        "precision": round(float(precision), 6),
        "false_positive_rate": round(float(false_positive_rate), 6),
        "cost_without_model": round(costs["cost_without_model"], 2),
        "cost_with_model": round(costs["cost_with_model"], 2),
        "review_cost": round(costs["review_cost"], 2),
        "operational_cost": round(costs["operational_cost"], 2),
        "operational_cost_total": round(costs["operational_cost_total"], 2),
        "net_savings": round(costs["net_savings"], 2),
        "roi": round(costs["roi"], 6),
        "value_per_1000_urls": round(costs["value_per_1000_urls"], 2),
        "annual_cost_without_model": round(float(annual_cost_without_model), 2),
        "annual_cost_with_model": round(float(annual_cost_with_model), 2),
        "annual_net_savings": round(float(annual_net_savings), 2),
        "value_test": round(costs["net_savings"], 2),
        "value_per_case": round(float(costs["net_savings"] / total), 4),
        "annual_value": round(float(annual_net_savings), 2),
    }


def threshold_search(
    y_true,
    y_proba_phishing,
    assumptions=None,
    start=0.01,
    stop=0.99,
    step=0.01,
    minimum_recall=None,
):
    """Search the threshold that maximizes net savings subject to minimum recall."""
    a = {**DEFAULT_BUSINESS_ASSUMPTIONS, **(assumptions or {})}
    min_recall = a["minimum_recall"] if minimum_recall is None else minimum_recall
    thresholds = np.arange(start, stop + step, step)
    rows = [
        evaluate_threshold(y_true, y_proba_phishing, threshold=t, assumptions=assumptions)
        for t in thresholds
    ]
    df = pd.DataFrame(rows)
    eligible = df[df["recall"] >= min_recall]
    if eligible.empty:
        eligible = df.copy()
    best = (
        eligible.sort_values(
            ["net_savings", "fn", "fp", "threshold"],
            ascending=[False, True, True, True],
        )
        .iloc[0]
        .to_dict()
    )
    return df, best


def evaluate_business_scenarios(
    y_true,
    y_proba_phishing,
    scenarios=None,
    start=0.01,
    stop=0.99,
    step=0.01,
):
    """Evaluate threshold policy for conservative, base and severe scenarios."""
    rows = []
    scenario_map = scenarios or BUSINESS_VALUE_SCENARIOS
    for scenario, assumptions in scenario_map.items():
        _, best = threshold_search(
            y_true,
            y_proba_phishing,
            assumptions=assumptions,
            start=start,
            stop=stop,
            step=step,
            minimum_recall=assumptions.get("minimum_recall"),
        )
        reference = evaluate_threshold(y_true, y_proba_phishing, threshold=0.5, assumptions=assumptions)
        rows.append({
            "scenario": scenario,
            "cost_false_negative": assumptions["cost_false_negative"],
            "cost_false_positive": assumptions["cost_false_positive"],
            "cost_review": assumptions["cost_review"],
            "cost_operational_per_url": assumptions["cost_operational_per_url"],
            "minimum_recall": assumptions["minimum_recall"],
            "recommended_threshold": best["threshold"],
            "reference_threshold": reference["threshold"],
            "recommended_recall": best["recall"],
            "recommended_fp": best["fp"],
            "recommended_fn": best["fn"],
            "recommended_net_savings": best["net_savings"],
            "reference_net_savings": reference["net_savings"],
            "incremental_savings_vs_050": round(
                float(best["net_savings"] - reference["net_savings"]), 2
            ),
            "recommended_roi": best["roi"],
            "recommended_value_per_1000_urls": best["value_per_1000_urls"],
            "annual_net_savings": best["annual_net_savings"],
        })
    return pd.DataFrame(rows)


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
