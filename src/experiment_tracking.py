from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np


def start_run(run_name, tags=None):
    """Start an MLflow run with project tags."""
    return mlflow.start_run(run_name=run_name, tags=tags or {})


def _is_loggable_value(value):
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def log_params(params):
    """Log parameters and return a list of keys that were accepted."""
    logged = []
    for key, value in (params or {}).items():
        if not _is_loggable_value(value):
            continue
        try:
            mlflow.log_param(str(key), value)
            logged.append(str(key))
        except Exception:
            continue
    return logged


def log_metrics(metrics):
    """Log numeric metrics and return a list of keys that were accepted."""
    logged = []
    for key, value in (metrics or {}).items():
        if not _is_loggable_value(value):
            continue
        try:
            mlflow.log_metric(str(key), float(value))
            logged.append(str(key))
        except Exception:
            continue
    return logged


def log_artifacts(paths, artifact_path=None):
    """Log existing files as MLflow artifacts without failing the full flow."""
    logged = []
    notes = []
    for path in paths or []:
        artifact = Path(path)
        if not artifact.exists():
            notes.append(f"missing_artifact:{artifact}")
            continue
        try:
            mlflow.log_artifact(str(artifact), artifact_path=artifact_path)
            logged.append(str(artifact))
        except Exception as exc:
            notes.append(f"artifact_error:{artifact}:{exc}")
    return logged, notes


def log_model_or_artifact(model_path, artifact_path="model"):
    """
    Try to load and register a sklearn-compatible model.

    If loading or MLflow model logging fails, the .pkl file is still registered as
    a regular artifact so the run keeps a traceable model reference.
    """
    path = Path(model_path)
    if not path.exists():
        return {
            "status": "missing",
            "artifacts_logged": [],
            "notes": [f"missing_model:{path}"],
        }

    try:
        model = joblib.load(path)
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
        return {
            "status": "registered_model",
            "artifacts_logged": [str(path)],
            "notes": ["model_logged_with_mlflow_sklearn"],
        }
    except Exception as exc:
        try:
            mlflow.log_artifact(str(path), artifact_path="model_artifact")
            return {
                "status": "artifact_only",
                "artifacts_logged": [str(path)],
                "notes": [f"model_load_or_log_error:{exc}"],
            }
        except Exception as artifact_exc:
            return {
                "status": "error",
                "artifacts_logged": [],
                "notes": [
                    f"model_load_or_log_error:{exc}",
                    f"model_artifact_error:{artifact_exc}",
                ],
            }
