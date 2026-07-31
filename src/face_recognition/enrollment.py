"""Build an identity database from consented reference photos."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import cv2 as cv
import numpy as np

from face_recognition.database import FaceDatabase
from face_recognition.detection import DetectedFace, Image, read_image
from face_recognition.embedding import Embedding, normalize
from face_recognition.errors import EnrollmentError

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
REFERENCE_MAX_DIMENSION = 800


class Detector(Protocol):
    def detect(self, image: Image) -> list[DetectedFace]: ...


class Embedder(Protocol):
    def embed(self, image: Image, face: DetectedFace) -> Embedding: ...


def reference_images(identity_directory: Path) -> list[Path]:
    return sorted(
        path
        for path in identity_directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def prepare_reference_image(
    image: Image, *, max_dimension: int = REFERENCE_MAX_DIMENSION
) -> Image:
    """Downscale large phone photos for stable YuNet detection and alignment."""
    height, width = image.shape[:2]
    largest_dimension = max(height, width)
    if largest_dimension <= max_dimension:
        return image
    scale = max_dimension / largest_dimension
    resized = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    return np.asarray(resized, dtype=np.uint8)


def enroll(
    references: Path,
    detector: Detector,
    embedder: Embedder,
    *,
    image_loader: Callable[[Path], Image] = read_image,
) -> FaceDatabase:
    if not references.is_dir():
        raise EnrollmentError(f"References directory not found: {references}")
    identity_directories = sorted(
        path for path in references.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    if not identity_directories:
        raise EnrollmentError("References directory contains no identity folders")

    labels: list[str] = []
    identity_embeddings: list[Embedding] = []
    expected_dimension: int | None = None

    for identity_directory in identity_directories:
        images = reference_images(identity_directory)
        if not images:
            raise EnrollmentError(f"No supported images for identity: {identity_directory.name}")
        samples: list[Embedding] = []
        for image_path in images:
            image = prepare_reference_image(image_loader(image_path))
            faces = detector.detect(image)
            if len(faces) != 1:
                raise EnrollmentError(
                    "Reference image must contain exactly one face; "
                    f"found {len(faces)}: {image_path}"
                )
            sample = embedder.embed(image, faces[0])
            if expected_dimension is None:
                expected_dimension = sample.size
            elif sample.size != expected_dimension:
                raise EnrollmentError("Face model returned inconsistent embedding dimensions")
            samples.append(sample)
        averaged = np.mean(np.stack(samples), axis=0, dtype=np.float32)
        labels.append(identity_directory.name)
        identity_embeddings.append(normalize(np.asarray(averaged, dtype=np.float32)))

    return FaceDatabase(
        labels=tuple(labels),
        embeddings=np.stack(identity_embeddings).astype(np.float32),
    )
