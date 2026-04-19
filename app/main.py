from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import Settings, get_settings
from app.inference import InferenceService
from app.schemas import HealthResponse, ModelInfoResponse, PredictResponse, RootResponse


def _read_image(contents: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(contents))
        image.load()
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.service = InferenceService(settings)
    yield


app = FastAPI(
    title=get_settings().app_name,
    version=get_settings().app_version,
    description="FastAPI service for plant disease image classification.",
    lifespan=lifespan,
)


def _service(request: Request) -> InferenceService:
    return request.app.state.service


def _settings(request: Request) -> Settings:
    return request.app.state.settings


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(
        service="Plant Disease Classifier API",
        docs_url="/docs",
        openapi_url="/openapi.json",
        health_url="/health",
        predict_url="/predict",
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = _service(request)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_path=service.metadata.checkpoint_path,
        device=service.metadata.device,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    metadata = _service(request).metadata
    return ModelInfoResponse(
        backbone=metadata.backbone,
        checkpoint_path=metadata.checkpoint_path,
        classes=metadata.class_names,
        device=metadata.device,
        dropout=metadata.dropout,
        image_size=metadata.image_size,
        mean=metadata.mean,
        num_classes=metadata.num_classes,
        source_type=metadata.source_type,
        std=metadata.std,
        tta_map=metadata.tta_map,
        val_map=metadata.val_map,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="Leaf image to classify")],
    top_k: Annotated[int | None, Form(ge=1)] = None,
) -> PredictResponse:
    settings = _settings(request)
    service = _service(request)

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload is too large. Limit is {settings.max_upload_size_mb} MB.",
        )

    resolved_top_k = top_k or settings.default_top_k
    if resolved_top_k > settings.max_top_k:
        raise HTTPException(
            status_code=400,
            detail=f"top_k must be less than or equal to {settings.max_top_k}.",
        )

    image = _read_image(contents)
    result = service.predict(image=image, top_k=resolved_top_k)

    return PredictResponse(
        filename=file.filename or "upload",
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        predictions=result["predictions"],
        inference_ms=result["inference_ms"],
        image_size=service.metadata.image_size,
        model_backbone=service.metadata.backbone,
        top_k=len(result["predictions"]),
    )
