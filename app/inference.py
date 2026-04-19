from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch import nn

from app.config import Settings


DEFAULT_BACKBONE = "efficientnet_b2"
DEFAULT_IMAGE_SIZE = 260
DEFAULT_IMAGENET_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class ModelMetadata:
    checkpoint_path: str
    source_type: str
    backbone: str
    num_classes: int
    image_size: int
    dropout: float
    class_names: list[str]
    mean: list[float]
    std: list[float]
    device: str
    val_map: float | None
    tta_map: float | None


class PlantDiseaseClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        in_features = getattr(self.backbone, "num_features", None)
        if not in_features:
            raise ValueError(f"Unable to determine num_features for backbone '{backbone_name}'.")

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        return self.head(features)


class InferenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = Lock()

        payload = self._load_payload(settings.model_path)
        metadata = self._build_metadata(payload, settings)
        state_dict = self._extract_state_dict(payload)

        self.model = PlantDiseaseClassifier(
            backbone_name=metadata.backbone,
            num_classes=metadata.num_classes,
            dropout=metadata.dropout,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        data_config = resolve_data_config(
            {
                "input_size": (3, metadata.image_size, metadata.image_size),
                "mean": tuple(metadata.mean),
                "std": tuple(metadata.std),
            },
            model=self.model.backbone,
        )
        self.transform = create_transform(**data_config, is_training=False)
        self.metadata = metadata

    def predict(self, image: Image.Image, top_k: int) -> dict[str, Any]:
        prepared = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        limited_top_k = min(max(1, top_k), self.metadata.num_classes)

        with self._lock, torch.inference_mode():
            started = perf_counter()
            logits = self.model(prepared)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
            elapsed_ms = round((perf_counter() - started) * 1000, 2)

        scores, indices = torch.topk(probabilities, k=limited_top_k)
        predictions = [
            {
                "label": self.metadata.class_names[index],
                "confidence": round(float(score), 6),
            }
            for score, index in zip(scores.tolist(), indices.tolist(), strict=False)
        ]

        best = predictions[0]
        return {
            "predicted_class": best["label"],
            "confidence": best["confidence"],
            "predictions": predictions,
            "inference_ms": elapsed_ms,
        }

    @staticmethod
    def _load_payload(model_path: Path) -> Any:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return torch.load(model_path, map_location="cpu", weights_only=False)

    def _build_metadata(self, payload: Any, settings: Settings) -> ModelMetadata:
        source_type = "bundle" if isinstance(payload, dict) and "state_dict" in payload else "state_dict"
        class_names = self._resolve_class_names(payload, settings.label_map_path)

        backbone = DEFAULT_BACKBONE
        image_size = DEFAULT_IMAGE_SIZE
        dropout = 0.0
        mean = DEFAULT_IMAGENET_MEAN
        std = DEFAULT_IMAGENET_STD
        val_map = None
        tta_map = None

        if source_type == "bundle":
            backbone = str(payload.get("backbone", backbone))
            image_size = int(payload.get("img_size", image_size))
            dropout = float(payload.get("dropout", dropout))
            mean = [float(value) for value in payload.get("imagenet_mean", mean)]
            std = [float(value) for value in payload.get("imagenet_std", std)]
            val_map = self._maybe_float(payload.get("val_mAP"))
            tta_map = self._maybe_float(payload.get("tta_mAP"))

        return ModelMetadata(
            checkpoint_path=str(settings.model_path),
            source_type=source_type,
            backbone=backbone,
            num_classes=len(class_names),
            image_size=image_size,
            dropout=dropout,
            class_names=class_names,
            mean=mean,
            std=std,
            device=str(self.device),
            val_map=val_map,
            tta_map=tta_map,
        )

    @staticmethod
    def _extract_state_dict(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict) and "state_dict" in payload:
            raw_state_dict = payload["state_dict"]
        elif isinstance(payload, dict):
            raw_state_dict = payload
        else:
            raise TypeError("Unsupported checkpoint format. Expected a state_dict or a bundle dict.")

        state_dict: dict[str, Any] = {}
        for key, value in raw_state_dict.items():
            normalized_key = key[7:] if key.startswith("module.") else key
            state_dict[normalized_key] = value
        return state_dict

    @staticmethod
    def _resolve_class_names(payload: Any, label_map_path: Path) -> list[str]:
        class_to_id = None
        if isinstance(payload, dict):
            class_to_id = payload.get("class_to_id")

        if isinstance(class_to_id, dict) and class_to_id:
            return [
                label
                for label, _ in sorted(class_to_id.items(), key=lambda item: int(item[1]))
            ]

        with label_map_path.open("r", encoding="utf-8") as handle:
            label_map = json.load(handle)

        id_to_class = label_map.get("id_to_class", {})
        if not isinstance(id_to_class, dict) or not id_to_class:
            raise ValueError(f"Unable to resolve class names from {label_map_path}.")

        return [label for _, label in sorted(id_to_class.items(), key=lambda item: int(item[0]))]

    @staticmethod
    def _maybe_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)
