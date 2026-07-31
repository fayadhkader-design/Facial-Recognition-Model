from pathlib import Path

import numpy as np
import pytest

from face_recognition.detection import DetectedFace
from face_recognition.enrollment import enroll, prepare_reference_image
from face_recognition.errors import EnrollmentError


class FakeDetector:
    def __init__(self, count: int = 1) -> None:
        self.count = count

    def detect(self, image):
        face = DetectedFace(0, 0, 1, 1, 1.0, np.zeros(15, dtype=np.float32))
        return [face] * self.count


class FakeEmbedder:
    def embed(self, image, face):
        return image[:2, 0, 0].astype(np.float32)


def loader(path: Path):
    value = 1.0 if "one" in path.name else 0.0
    return np.array([[[value]], [[1.0 - value]]], dtype=np.float32)


def test_enrolls_sorted_identities_and_averages_samples(tmp_path: Path) -> None:
    (tmp_path / "Bob").mkdir()
    (tmp_path / "Alice").mkdir()
    (tmp_path / "Alice" / "one.jpg").touch()
    (tmp_path / "Alice" / "two.png").touch()
    (tmp_path / "Bob" / "one.jpeg").touch()

    database = enroll(tmp_path, FakeDetector(), FakeEmbedder(), image_loader=loader)

    assert database.labels == ("Alice", "Bob")
    np.testing.assert_allclose(database.embeddings[0], [2**-0.5, 2**-0.5])
    np.testing.assert_allclose(database.embeddings[1], [1.0, 0.0])


@pytest.mark.parametrize("count", [0, 2])
def test_enrollment_requires_exactly_one_face(tmp_path: Path, count: int) -> None:
    (tmp_path / "Alice").mkdir()
    (tmp_path / "Alice" / "one.jpg").touch()

    with pytest.raises(EnrollmentError, match=f"found {count}"):
        enroll(tmp_path, FakeDetector(count), FakeEmbedder(), image_loader=loader)


def test_enrollment_rejects_empty_identity_folder(tmp_path: Path) -> None:
    (tmp_path / "Alice").mkdir()

    with pytest.raises(EnrollmentError, match="No supported images"):
        enroll(tmp_path, FakeDetector(), FakeEmbedder(), image_loader=loader)


def test_large_reference_image_is_downscaled() -> None:
    image = np.zeros((1600, 1200, 3), dtype=np.uint8)

    resized = prepare_reference_image(image)

    assert resized.shape == (800, 600, 3)
