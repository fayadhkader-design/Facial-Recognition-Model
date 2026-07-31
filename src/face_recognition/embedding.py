"""SFace alignment and embedding adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from face_recognition.detection import DetectedFace, Image
from face_recognition.errors import ImageError, ModelError

Embedding = NDArray[np.float32]


class RecognizerBackend(Protocol):
    def alignCrop(self, image: Image, face_row: NDArray[np.float32]) -> Image: ...

    def feature(self, aligned_face: Image) -> NDArray[np.float32]: ...


def normalize(vector: NDArray[np.float32]) -> Embedding:
    flattened = np.asarray(vector, dtype=np.float32).reshape(-1)
    magnitude = float(np.linalg.norm(flattened))
    if not np.isfinite(magnitude) or magnitude <= 0:
        raise ImageError("Face model returned an invalid embedding")
    return np.asarray(flattened / magnitude, dtype=np.float32)


class FaceEmbedder:
    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise ModelError(f"Face recognition model not found: {model_path}")
        try:
            backend = cv.FaceRecognizerSF.create(str(model_path), "")
        except cv.error as exc:
            raise ModelError(f"Could not load face recognition model: {model_path}") from exc
        self._recognizer = cast(RecognizerBackend, backend)

    def embed(self, image: Image, face: DetectedFace) -> Embedding:
        try:
            aligned = self._recognizer.alignCrop(image, face.row)
            features = self._recognizer.feature(aligned)
        except cv.error as exc:
            raise ImageError("Could not align or embed a detected face") from exc
        return normalize(features)
