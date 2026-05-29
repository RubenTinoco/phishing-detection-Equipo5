from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "final_model.pkl"
BUSINESS_VALUE_SUMMARY_PATH = ROOT / "reports" / "business_value_summary.csv"
REFERENCE_DATA_PATH = ROOT / "data" / "processed" / "test.csv"
TARGET_COL = "Result"
PHISHING_MODEL_LABELS = (-1, 0)
PHISHING_OUTPUT_LABEL = 1
LEGITIMATE_OUTPUT_LABEL = 0


def resolve_model_path() -> Path:
    configured = os.getenv("MODEL_PATH")
    if not configured:
        return DEFAULT_MODEL_PATH
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


def load_default_threshold() -> float:
    if BUSINESS_VALUE_SUMMARY_PATH.exists():
        try:
            summary = pd.read_csv(BUSINESS_VALUE_SUMMARY_PATH)
            values = dict(zip(summary["item"], summary["value"]))
            threshold = values.get("umbral_recomendado", values.get("umbral_optimo"))
            if threshold is not None:
                return float(threshold)
        except Exception:
            pass

    configured = os.getenv("DEFAULT_THRESHOLD")
    if configured:
        try:
            return float(configured)
        except ValueError:
            pass

    return 0.5


def load_expected_features(model: Any | None = None) -> list[str]:
    if model is not None and hasattr(model, "feature_names_in_"):
        return [str(col) for col in model.feature_names_in_]

    if REFERENCE_DATA_PATH.exists():
        columns = list(pd.read_csv(REFERENCE_DATA_PATH, nrows=0).columns)
        return [col for col in columns if col != TARGET_COL]

    return []


@lru_cache(maxsize=1)
def get_model_bundle():
    model_path = resolve_model_path()
    if not model_path.exists():
        return {
            "status": "missing",
            "model": None,
            "model_path": str(model_path),
            "error": f"model file not found: {model_path}",
            "expected_features": load_expected_features(None),
        }

    try:
        model = joblib.load(model_path)
        return {
            "status": "loaded",
            "model": model,
            "model_path": str(model_path),
            "error": None,
            "expected_features": load_expected_features(model),
        }
    except Exception as exc:
        return {
            "status": "error",
            "model": None,
            "model_path": str(model_path),
            "error": str(exc),
            "expected_features": load_expected_features(None),
        }


def phishing_probability(model, features: pd.DataFrame):
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    phishing_label = next((label for label in PHISHING_MODEL_LABELS if label in classes), None)
    if phishing_label is None:
        return None
    return probabilities[:, classes.index(phishing_label)]


def label_from_probability(probability: float, threshold: float) -> tuple[int, str]:
    if probability >= threshold:
        return PHISHING_OUTPUT_LABEL, "phishing"
    return LEGITIMATE_OUTPUT_LABEL, "legitimo"


def label_from_model_prediction(prediction: Any) -> tuple[int, str]:
    if prediction in PHISHING_MODEL_LABELS:
        return PHISHING_OUTPUT_LABEL, "phishing"
    return LEGITIMATE_OUTPUT_LABEL, "legitimo"
