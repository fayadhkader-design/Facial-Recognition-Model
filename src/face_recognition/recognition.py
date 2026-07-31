"""Recognize and annotate faces in a group image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from face_recognition.database import FaceDatabase
from face_recognition.detection import DetectedFace, Image
from face_recognition.enrollment import Detector, Embedder
from face_recognition.errors import ImageError
from face_recognition.matching import IdentityMatch, match_identity


@dataclass(frozen=True, slots=True)
class RecognizedFace:
    face: DetectedFace
    match: IdentityMatch

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.match.label,
            "known": self.match.known,
            "similarity": round(self.match.similarity, 6),
            "box": {
                "x": self.face.x,
                "y": self.face.y,
                "width": self.face.width,
                "height": self.face.height,
            },
        }


def recognize_faces(
    image: Image,
    database: FaceDatabase,
    detector: Detector,
    embedder: Embedder,
    *,
    threshold: float,
) -> list[RecognizedFace]:
    return [
        RecognizedFace(
            face=face,
            match=match_identity(embedder.embed(image, face), database, threshold=threshold),
        )
        for face in detector.detect(image)
    ]


def annotate_image(image: Image, results: list[RecognizedFace]) -> Image:
    """Return a labeled copy, leaving the caller's input untouched."""
    annotated = image.copy()
    for result in results:
        x, y, width, height = result.face.box
        color = (48, 190, 80) if result.match.known else (50, 90, 230)
        end = (min(x + width, annotated.shape[1] - 1), min(y + height, annotated.shape[0] - 1))
        cv.rectangle(annotated, (x, y), end, color, 2)
        text = f"{result.match.label} {result.match.similarity:.3f}"
        text_y = max(18, y - 7)
        cv.putText(annotated, text, (x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return np.asarray(annotated, dtype=np.uint8)


def save_image(image: Image, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(output), image):
        raise ImageError(f"Could not save annotated image: {output}")

