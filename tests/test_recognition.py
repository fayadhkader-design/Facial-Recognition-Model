from pathlib import Path

import numpy as np

from face_recognition.database import FaceDatabase
from face_recognition.detection import DetectedFace
from face_recognition.matching import IdentityMatch
from face_recognition.recognition import RecognizedFace, annotate_image, recognize_faces, save_image


class FakeDetector:
    def detect(self, image):
        return [DetectedFace(5, 5, 10, 10, 0.99, np.zeros(15, dtype=np.float32))]


class FakeEmbedder:
    def embed(self, image, face):
        return np.array([1.0, 0.0], dtype=np.float32)


def test_recognizes_each_detected_face() -> None:
    database = FaceDatabase(("Alice",), np.array([[1.0, 0.0]], dtype=np.float32))

    results = recognize_faces(
        np.zeros((30, 30, 3), dtype=np.uint8),
        database,
        FakeDetector(),
        FakeEmbedder(),
        threshold=0.5,
    )

    assert results[0].match.label == "Alice"
    assert results[0].as_dict()["box"] == {"x": 5, "y": 5, "width": 10, "height": 10}


def test_annotation_does_not_modify_input() -> None:
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    original = image.copy()
    result = RecognizedFace(
        DetectedFace(5, 20, 20, 15, 0.99, np.zeros(15, dtype=np.float32)),
        IdentityMatch("Alice", 0.9, True),
    )

    annotated = annotate_image(image, [result])

    np.testing.assert_array_equal(image, original)
    assert np.any(annotated != original)


def test_save_image_creates_parent_directory(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "result.png"

    save_image(np.zeros((10, 10, 3), dtype=np.uint8), output)

    assert output.is_file()
