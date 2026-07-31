"""YuNet face detection adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from face_recognition.errors import ImageError, ModelError

FaceRow = NDArray[np.float32]
Image = NDArray[np.uint8]


class DetectorBackend(Protocol):
    def setInputSize(self, input_size: tuple[int, int]) -> None: ...

    def detect(self, image: Image) -> tuple[int, NDArray[np.float32] | None]: ...


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """A bounding box plus the raw YuNet landmarks required for alignment."""

    x: int
    y: int
    width: int
    height: int
    confidence: float
    row: FaceRow

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


class FaceDetector:
    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        if not model_path.is_file():
            raise ModelError(f"Face detector model not found: {model_path}")
        try:
            backend = cv.FaceDetectorYN.create(
                str(model_path), "", (320, 320), confidence_threshold, nms_threshold, top_k
            )
        except cv.error as exc:
            raise ModelError(f"Could not load face detector model: {model_path}") from exc
        self._detector = cast(DetectorBackend, backend)

    def detect(self, image: Image) -> list[DetectedFace]:
        if image.size == 0 or image.ndim != 3:
            raise ImageError("Expected a non-empty color image")
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, rows = self._detector.detect(image)
        if rows is None:
            return []
        return [
            DetectedFace(
                x=max(0, int(row[0])),
                y=max(0, int(row[1])),
                width=max(0, int(row[2])),
                height=max(0, int(row[3])),
                confidence=float(row[-1]),
                row=np.asarray(row, dtype=np.float32),
            )
            for row in rows
        ]


def read_image(path: Path) -> Image:
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    if image is None:
        raise ImageError(f"Could not read image: {path}")
    return np.asarray(image, dtype=np.uint8)
