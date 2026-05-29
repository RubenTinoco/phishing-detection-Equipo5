from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.model_loader import (
    TARGET_COL,
    get_model_bundle,
    label_from_model_prediction,
    label_from_probability,
    load_default_threshold,
    phishing_probability,
)
from api.schemas import PredictionRequest, PredictionResponse


PROJECT_NAME = "phishing-detection"

app = FastAPI(
    title="Phishing Detection API",
    version="1.0.0",
    description="Sprint 6 REST API for phishing detection inference.",
)


@app.get("/health")
def health():
    bundle = get_model_bundle()
    return {
        "project": PROJECT_NAME,
        "api_status": "ok",
        "model_status": bundle["status"],
        "model_path": bundle["model_path"],
        "model_error": bundle["error"],
        "expected_features_count": len(bundle["expected_features"]),
    }


def build_feature_frame(records: list[dict], expected_features: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    if df.empty:
        raise HTTPException(status_code=400, detail="records did not contain usable feature columns")

    if expected_features:
        missing = [col for col in expected_features if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "input records are missing required feature columns",
                    "missing_columns": missing,
                    "expected_columns": expected_features,
                },
            )
        df = df[expected_features]

    try:
        return df.apply(pd.to_numeric)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"all feature values must be numeric or numeric-compatible: {exc}",
        ) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    bundle = get_model_bundle()
    if bundle["status"] != "loaded" or bundle["model"] is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "model is not available",
                "model_status": bundle["status"],
                "model_path": bundle["model_path"],
                "model_error": bundle["error"],
            },
        )

    threshold = request.threshold if request.threshold is not None else load_default_threshold()
    features = build_feature_frame(request.records, bundle["expected_features"])
    model = bundle["model"]

    try:
        probabilities = phishing_probability(model, features)
        predictions = []
        if probabilities is not None:
            for probability in probabilities:
                prediction, label = label_from_probability(float(probability), threshold)
                predictions.append({
                    "prediction": prediction,
                    "prediction_label": label,
                    "phishing_probability": round(float(probability), 6),
                })
        else:
            raw_predictions = model.predict(features)
            for raw_prediction in raw_predictions:
                prediction, label = label_from_model_prediction(raw_prediction)
                predictions.append({
                    "prediction": prediction,
                    "prediction_label": label,
                    "phishing_probability": None,
                })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed with the loaded model: {exc}",
        ) from exc

    return {
        "model_status": bundle["status"],
        "threshold": round(float(threshold), 4),
        "n_records": len(predictions),
        "predictions": predictions,
    }
