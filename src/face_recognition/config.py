"""Typed runtime configuration and path defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_COSINE_THRESHOLD = 0.5
DEFAULT_DETECTION_THRESHOLD = 0.9
MODEL_DIRECTORY_ENV = "FACE_RECOGNITION_MODEL_DIR"


def default_model_directory() -> Path:
    """Return a user-overridable, repository-independent model directory."""
    configured = os.environ.get(MODEL_DIRECTORY_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / "models"


@dataclass(frozen=True, slots=True)
class ModelPaths:
    """Paths to the two OpenCV ONNX models."""

    detector: Path
    recognizer: Path

    @classmethod
    def in_directory(cls, directory: Path | None = None) -> ModelPaths:
        root = directory or default_model_directory()
        return cls(
            detector=root / "face_detection_yunet_2023mar.onnx",
            recognizer=root / "face_recognition_sface_2021dec.onnx",
        )
