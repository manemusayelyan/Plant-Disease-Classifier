from __future__ import annotations

from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    service: str
    docs_url: str
    openapi_url: str
    health_url: str
    predict_url: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str


class PredictionItem(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    predictions: list[PredictionItem]
    inference_ms: float
    image_size: int
    model_backbone: str
    top_k: int


class ModelInfoResponse(BaseModel):
    backbone: str
    checkpoint_path: str
    classes: list[str]
    device: str
    dropout: float
    image_size: int
    mean: list[float]
    num_classes: int
    source_type: str
    std: list[float]
    tta_map: float | None
    val_map: float | None
