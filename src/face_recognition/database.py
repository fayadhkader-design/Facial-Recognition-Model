"""Safe, versioned storage for labels and face embeddings."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from face_recognition.errors import DatabaseError

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FaceDatabase:
    labels: tuple[str, ...]
    embeddings: NDArray[np.float32]

    def __post_init__(self) -> None:
        if not self.labels:
            raise DatabaseError("Database must contain at least one identity")
        if len(set(self.labels)) != len(self.labels):
            raise DatabaseError("Database labels must be unique")
        if any(not label.strip() for label in self.labels):
            raise DatabaseError("Database labels cannot be empty")
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.labels):
            raise DatabaseError("Embedding matrix shape does not match labels")
        if self.embeddings.shape[1] == 0 or not np.all(np.isfinite(self.embeddings)):
            raise DatabaseError("Embedding matrix contains invalid values")
        norms = np.linalg.norm(self.embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise DatabaseError("All database embeddings must be normalized")


def save_database(database: FaceDatabase, path: Path) -> None:
    """Write a database atomically without object arrays or pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(
                stream,
                schema_version=np.array(SCHEMA_VERSION, dtype=np.int64),
                labels=np.asarray(database.labels, dtype=np.str_),
                embeddings=np.asarray(database.embeddings, dtype=np.float32),
            )
        temporary.replace(path)
    except OSError as exc:
        raise DatabaseError(f"Could not save database {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_database(path: Path) -> FaceDatabase:
    if not path.is_file():
        raise DatabaseError(f"Database not found: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"schema_version", "labels", "embeddings"}
            if set(archive.files) != required:
                raise DatabaseError("Database has missing or unexpected fields")
            version = int(archive["schema_version"])
            if version != SCHEMA_VERSION:
                raise DatabaseError(f"Unsupported database schema version: {version}")
            labels_array = archive["labels"]
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            if labels_array.ndim != 1 or labels_array.dtype.kind not in {"U", "S"}:
                raise DatabaseError("Database labels must be a one-dimensional text array")
            labels = tuple(str(label) for label in labels_array.tolist())
        return FaceDatabase(labels=labels, embeddings=embeddings)
    except DatabaseError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise DatabaseError(f"Could not safely load database {path}: {exc}") from exc

