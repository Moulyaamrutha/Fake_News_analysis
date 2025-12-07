from pydantic import BaseModel
from typing import List, Optional


class PredictRequest(BaseModel):
    text: str
    model_name: Optional[str] = None
    threshold: Optional[float] = 0.5


class BatchPredictRequest(BaseModel):
    texts: List[str]
    model_name: Optional[str] = None
    threshold: Optional[float] = 0.5


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model: str
