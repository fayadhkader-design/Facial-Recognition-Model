import numpy as np
import pytest

from face_recognition.database import FaceDatabase
from face_recognition.errors import FaceRecognitionError
from face_recognition.matching import match_identity


@pytest.fixture
def database() -> FaceDatabase:
    return FaceDatabase(
        labels=("Alice", "Bob"),
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_match_returns_closest_identity(database: FaceDatabase) -> None:
    match = match_identity(np.array([0.9, 0.1], dtype=np.float32), database, threshold=0.5)

    assert match.label == "Alice"
    assert match.known is True
    assert match.similarity > 0.9


def test_match_marks_score_below_threshold_unknown(database: FaceDatabase) -> None:
    match = match_identity(
        np.array([2**-0.5, 2**-0.5], dtype=np.float32), database, threshold=0.8
    )

    assert match.label == "Unknown"
    assert match.known is False


def test_match_includes_exact_threshold(database: FaceDatabase) -> None:
    match = match_identity(np.array([1.0, 0.0], dtype=np.float32), database, threshold=1.0)

    assert match.label == "Alice"


def test_match_rejects_dimension_mismatch(database: FaceDatabase) -> None:
    with pytest.raises(FaceRecognitionError, match="dimensions"):
        match_identity(np.array([1.0, 0.0, 0.0], dtype=np.float32), database, threshold=0.3)


def test_match_rejects_invalid_threshold(database: FaceDatabase) -> None:
    with pytest.raises(FaceRecognitionError, match="between -1 and 1"):
        match_identity(np.array([1.0, 0.0], dtype=np.float32), database, threshold=1.1)
