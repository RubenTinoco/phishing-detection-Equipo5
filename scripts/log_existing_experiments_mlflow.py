from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlflow_config import setup_mlflow
from src.experiment_tracking import (
    log_artifacts,
    log_metrics,
    log_model_or_artifact,
    log_params,
    start_run,
)


MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SUMMARY_PATH = REPORTS_DIR / "mlflow_tracking_summary.csv"

PROJECT_TAGS = {
    "project": "phishing-detection",
    "source": "existing_artifacts",
}

COMMON_ARTIFACTS = [
    MODELS_DIR / "experiments_log.csv",
    MODELS_DIR / "selection_decisions.csv",
]

FIGURE_ARTIFACTS = [
    FIGURES_DIR / "confusion_matrix_final.png",
    FIGURES_DIR / "roc_curve_final.png",
    FIGURES_DIR / "pr_curve_final.png",
    FIGURES_DIR / "final_comparison.png",
    FIGURES_DIR / "business_value_thresholds.png",
    FIGURES_DIR / "business_value_scenarios.png",
]

BUSINESS_ARTIFACTS = [
    REPORTS_DIR / "business_value_summary.csv",
    REPORTS_DIR / "business_value_thresholds.csv",
]


MODEL_NAME_BY_STEM = {
    "baseline_logistic_regression": "Logistic Regression",
    "baseline_decision_tree": "Decision Tree",
    "baseline_random_forest": "Random Forest",
    "baseline_gradient_boosting": "Gradient Boosting",
    "baseline_svm": "SVM",
    "baseline_knn": "KNN",
    "baseline_naive_bayes": "Naive Bayes",
    "tuned_random_forest": "Random Forest",
    "tuned_gradient_boosting": "Gradient Boosting",
    "tuned_logistic_regression": "Logistic Regression",
    "tuned_xgboost": "XGBoost",
    "tuned_lgbm": "LightGBM",
    "tuned_stacking": "Stacking",
    "final_model": "Modelo Final",
}


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def numeric(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def normalize_metric_name(name):
    aliases = {
        "Accuracy": "accuracy",
        "accuracy_cv_mean": "accuracy",
        "Precision": "precision",
        "precision_cv_mean": "precision",
        "Recall": "recall",
        "recall_cv_mean": "recall",
        "F1": "f1",
        "f1_cv_mean": "f1",
        "AUC-ROC": "roc_auc",
        "auc_cv_mean": "roc_auc",
        "PR-AUC": "pr_auc",
        "pr_auc": "pr_auc",
        "best_score": "f1",
    }
    return aliases.get(name, name)


def row_to_metrics(row, columns):
    metrics = {}
    for column in columns:
        if column in row:
            value = numeric(row[column])
            if value is not None:
                metrics[normalize_metric_name(column)] = value
    return metrics


def business_summary_values():
    summary = read_csv(REPORTS_DIR / "business_value_summary.csv")
    if summary.empty or not {"item", "value"}.issubset(summary.columns):
        return {}
    return dict(zip(summary["item"], summary["value"]))


def baseline_metrics(model_name, baseline_df):
    if baseline_df.empty or "Modelo" not in baseline_df.columns:
        return {}
    rows = baseline_df[baseline_df["Modelo"] == model_name]
    if rows.empty:
        return {}
    return row_to_metrics(
        rows.iloc[0],
        ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "PR-AUC"],
    )


def tuned_metrics(model_stem, experiments_df, ensemble_df):
    model_key = model_stem.replace("tuned_", "")
    metrics = {}

    if model_key in {"xgboost", "lgbm", "stacking"} and not ensemble_df.empty:
        lookup = {
            "xgboost": "XGBoost",
            "lgbm": "LightGBM",
            "stacking": "Stacking",
        }
        target = lookup[model_key]
        rows = ensemble_df[ensemble_df["Modelo"].astype(str).str.contains(target, case=False, na=False)]
        if not rows.empty:
            metrics.update(row_to_metrics(rows.iloc[0], ["Precision", "Recall", "F1", "AUC-ROC", "PR-AUC"]))

    if not experiments_df.empty and "model" in experiments_df.columns:
        rows = experiments_df[experiments_df["model"].astype(str).str.lower() == model_key]
        if not rows.empty:
            row = rows.iloc[-1]
            metrics.update(row_to_metrics(row, ["best_score", "time_seconds"]))

    return metrics


def final_metrics(final_df, business_values):
    metrics = {}
    if not final_df.empty:
        label_col = final_df.columns[0]
        rows = final_df[final_df[label_col] == "Modelo Final"]
        if not rows.empty:
            metrics.update(row_to_metrics(rows.iloc[0], ["Precision", "Recall", "F1", "AUC-ROC", "PR-AUC"]))

    business_metric_map = {
        "umbral_recomendado": "threshold",
        "ahorro_neto_recomendado_usd": "ahorro_neto",
        "roi_recomendado": "roi",
        "valor_por_1000_urls_recomendado_usd": "valor_por_1000_urls",
        "ahorro_incremental_vs_050_usd": "ahorro_incremental_vs_050",
    }
    for source_key, metric_key in business_metric_map.items():
        value = numeric(business_values.get(source_key))
        if value is not None:
            metrics[metric_key] = value
    return metrics


def tuned_params(model_stem, experiments_df):
    model_key = model_stem.replace("tuned_", "")
    if experiments_df.empty or "model" not in experiments_df.columns:
        return {}
    rows = experiments_df[experiments_df["model"].astype(str).str.lower() == model_key]
    if rows.empty:
        return {}
    row = rows.iloc[-1]
    params = {"model_path": row.get("path")}
    if "method" in row and pd.notna(row["method"]):
        params["method"] = row["method"]
    if "params" in row and pd.notna(row["params"]):
        params["best_params"] = row["params"]
    return params


def run_type_for_model(model_path):
    stem = model_path.stem
    if stem.startswith("baseline_"):
        return "baseline", "Sprint 3"
    if stem == "final_model":
        return "final", "Sprint 4"
    if stem in {"tuned_xgboost", "tuned_lgbm", "tuned_stacking"}:
        return "ensemble", "Sprint 4"
    return "tuned", "Sprint 4"


def artifacts_for_run(run_type):
    if run_type == "baseline":
        return COMMON_ARTIFACTS + [
            MODELS_DIR / "baseline_comparison.csv",
            MODELS_DIR / "baseline_f1_comparison.png",
            MODELS_DIR / "confusion_matrices.png",
            MODELS_DIR / "roc_curves.png",
            MODELS_DIR / "precision_recall_curves.png",
        ]
    if run_type in {"tuned", "ensemble"}:
        return COMMON_ARTIFACTS + [
            MODELS_DIR / "ensemble_comparison.csv",
            MODELS_DIR / "final_comparison.csv",
            FIGURES_DIR / "tuning_comparison.png",
            FIGURES_DIR / "ensemble_comparison.png",
            FIGURES_DIR / "final_comparison.png",
        ]
    return COMMON_ARTIFACTS + BUSINESS_ARTIFACTS + FIGURE_ARTIFACTS + [
        MODELS_DIR / "final_comparison.csv",
        FIGURES_DIR / "final_validation_curves.png",
    ]


def build_run_specs():
    baseline_df = read_csv(MODELS_DIR / "baseline_comparison.csv")
    experiments_df = read_csv(MODELS_DIR / "experiments_log.csv")
    ensemble_df = read_csv(MODELS_DIR / "ensemble_comparison.csv")
    final_df = read_csv(MODELS_DIR / "final_comparison.csv")
    business_values = business_summary_values()

    model_paths = sorted(MODELS_DIR.glob("baseline_*.pkl"))
    model_paths += sorted(MODELS_DIR.glob("tuned_*.pkl"))
    final_model = MODELS_DIR / "final_model.pkl"
    if final_model.exists():
        model_paths.append(final_model)

    specs = []
    for model_path in model_paths:
        run_type, sprint = run_type_for_model(model_path)
        model_name = MODEL_NAME_BY_STEM.get(model_path.stem, model_path.stem)
        params = {"model_path": str(model_path.relative_to(ROOT))}
        metrics = {}

        if run_type == "baseline":
            metrics = baseline_metrics(model_name, baseline_df)
        elif run_type in {"tuned", "ensemble"}:
            metrics = tuned_metrics(model_path.stem, experiments_df, ensemble_df)
            params.update(tuned_params(model_path.stem, experiments_df))
        elif run_type == "final":
            metrics = final_metrics(final_df, business_values)
            params["threshold_source"] = "reports/business_value_summary.csv"

        specs.append({
            "model_path": model_path,
            "model_name": model_name,
            "run_type": run_type,
            "sprint": sprint,
            "params": params,
            "metrics": metrics,
            "artifacts": artifacts_for_run(run_type),
        })
    return specs


def log_run(spec):
    tags = {
        **PROJECT_TAGS,
        "sprint": spec["sprint"],
        "run_type": spec["run_type"],
    }
    run_name = f"{spec['sprint']} - {spec['run_type']} - {spec['model_name']}"
    notes = []
    artifacts_logged = []
    status = "success"

    with start_run(run_name=run_name, tags=tags) as run:
        log_params({
            "model_name": spec["model_name"],
            "run_type": spec["run_type"],
            "sprint": spec["sprint"],
            **spec["params"],
        })
        metrics_logged = log_metrics(spec["metrics"])
        model_result = log_model_or_artifact(spec["model_path"])
        artifacts_logged.extend(model_result["artifacts_logged"])
        notes.extend(model_result["notes"])

        artifact_list, artifact_notes = log_artifacts(spec["artifacts"], artifact_path="project_artifacts")
        artifacts_logged.extend(artifact_list)
        notes.extend(artifact_notes)

        if model_result["status"] in {"artifact_only", "missing"}:
            status = "partial"
        if model_result["status"] == "error":
            status = "error"

        return {
            "run_id": run.info.run_id,
            "model_name": spec["model_name"],
            "run_type": spec["run_type"],
            "sprint": spec["sprint"],
            "status": status,
            "metrics_logged": ",".join(metrics_logged),
            "artifacts_logged": str(len(artifacts_logged)),
            "notes": " | ".join(notes),
        }


def main():
    setup_mlflow()
    specs = build_run_specs()
    rows = [log_run(spec) for spec in specs]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)
    print(f"Registered {len(rows)} MLflow runs.")
    print(f"Summary written to {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
