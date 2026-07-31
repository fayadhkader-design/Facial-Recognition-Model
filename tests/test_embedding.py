from pathlib import Path

import numpy as np
import pytest

from face_recognition.detection import DetectedFace
from face_recognition.embedding import FaceEmbedder, normalize
from face_recognition.errors import ImageError, ModelError


class FakeRecognizer:
    def alignCrop(self, image, face_row):
        return image

    def feature(self, aligned_face):
        return np.array([[3.0, 4.0]], dtype=np.float32)


def sample_face() -> DetectedFace:
    return DetectedFace(0, 0, 10, 10, 0.99, np.zeros(15, dtype=np.float32))


def test_normalize_returns_unit_vector() -> None:
    result = normalize(np.array([3.0, 4.0], dtype=np.float32))

    np.testing.assert_allclose(result, [0.6, 0.8])
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_normalize_rejects_zero_vector() -> None:
    with pytest.raises(ImageError, match="invalid embedding"):
        normalize(np.zeros(2, dtype=np.float32))


def test_embed_aligns_and_normalizes() -> None:
    embedder = FaceEmbedder.__new__(FaceEmbedder)
    embedder._recognizer = FakeRecognizer()

    result = embedder.embed(np.zeros((10, 10, 3), dtype=np.uint8), sample_face())

    np.testing.assert_allclose(result, [0.6, 0.8])


def test_embedder_requires_model_file(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="not found"):
        FaceEmbedder(tmp_path / "missing.onnx")
