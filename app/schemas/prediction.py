from pydantic import BaseModel
from typing import Dict


class PredictionData(BaseModel):
    filename: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]


class PredictionResponse(BaseModel):
    success: bool
    data: PredictionData