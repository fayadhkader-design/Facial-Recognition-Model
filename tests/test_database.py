from pathlib import Path

import numpy as np
import pytest

from face_recognition.database import FaceDatabase, load_database, save_database
from face_recognition.errors import DatabaseError


def sample_database() -> FaceDatabase:
    return FaceDatabase(
        labels=("Alice", "Bob"),
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_database_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "faces.npz"

    save_database(sample_database(), path)
    loaded = load_database(path)

    assert loaded.labels == ("Alice", "Bob")
    np.testing.assert_allclose(loaded.embeddings, sample_database().embeddings)


def test_database_rejects_duplicate_labels() -> None:
    with pytest.raises(DatabaseError, match="unique"):
        FaceDatabase(
            labels=("Alice", "Alice"),
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )


def test_database_rejects_unnormalized_embeddings() -> None:
    with pytest.raises(DatabaseError, match="normalized"):
        FaceDatabase(labels=("Alice",), embeddings=np.array([[2.0, 0.0]], dtype=np.float32))


def test_database_rejects_pickle_backed_labels(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.npz"
    np.savez(
        path,
        schema_version=np.array(1),
        labels=np.array([{"name": "Alice"}], dtype=object),
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(DatabaseError, match="safely load"):
        load_database(path)


def test_database_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "future.npz"
    np.savez(
        path,
        schema_version=np.array(99),
        labels=np.array(["Alice"]),
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(DatabaseError, match="schema version"):
        load_database(path)
