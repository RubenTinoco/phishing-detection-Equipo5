from typing import Any

from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., description="Feature rows to score.")
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @validator("records")
    def records_must_not_be_empty(cls, value):
        if not value:
            raise ValueError("records must contain at least one row")
        if not all(isinstance(row, dict) for row in value):
            raise ValueError("each record must be a JSON object")
        return value


class PredictionItem(BaseModel):
    prediction: int
    prediction_label: str
    phishing_probability: float | None = None


class PredictionResponse(BaseModel):
    model_status: str
    threshold: float
    n_records: int
    predictions: list[PredictionItem]
