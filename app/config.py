from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "checkpoints" / "plantdiseaseclassifier_efficientnet_b2_bundle.pt"
DEFAULT_LABEL_MAP_PATH = ROOT_DIR / "prepared" / "label_map.json"


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    model_path: Path
    label_map_path: Path
    default_top_k: int
    max_top_k: int
    max_upload_size_mb: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "Plant Disease Classifier API"),
        app_version=os.getenv("APP_VERSION", "1.0.0"),
        model_path=Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)),
        label_map_path=Path(os.getenv("LABEL_MAP_PATH", DEFAULT_LABEL_MAP_PATH)),
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
        max_top_k=int(os.getenv("MAX_TOP_K", "10")),
        max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "10")),
    )
