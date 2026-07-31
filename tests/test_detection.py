from pathlib import Path

import numpy as np
import pytest

from face_recognition.detection import FaceDetector
from face_recognition.errors import ImageError, ModelError


class FakeBackend:
    def __init__(self, rows):
        self.rows = rows
        self.size = None

    def setInputSize(self, input_size):
        self.size = input_size

    def detect(self, image):
        return 1, self.rows


def detector_with(backend: FakeBackend) -> FaceDetector:
    detector = FaceDetector.__new__(FaceDetector)
    detector._detector = backend
    return detector


def test_detector_converts_yunet_rows() -> None:
    row = np.array([-2, 5, 30, 40, *range(10), 0.95], dtype=np.float32)
    backend = FakeBackend(np.array([row]))

    faces = detector_with(backend).detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert backend.size == (200, 100)
    assert faces[0].box == (0, 5, 30, 40)
    assert faces[0].confidence == pytest.approx(0.95)


def test_detector_returns_empty_list() -> None:
    assert detector_with(FakeBackend(None)).detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_detector_rejects_empty_image() -> None:
    with pytest.raises(ImageError, match="non-empty color image"):
        detector_with(FakeBackend(None)).detect(np.array([], dtype=np.uint8))


def test_detector_requires_model_file(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="not found"):
        FaceDetector(tmp_path / "missing.onnx")
