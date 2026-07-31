"""Cosine-similarity identity matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from face_recognition.database import FaceDatabase
from face_recognition.embedding import Embedding, normalize
from face_recognition.errors import FaceRecognitionError


@dataclass(frozen=True, slots=True)
class IdentityMatch:
    label: str
    similarity: float
    known: bool


def match_identity(
    embedding: Embedding,
    database: FaceDatabase,
    *,
    threshold: float,
) -> IdentityMatch:
    if not -1.0 <= threshold <= 1.0:
        raise FaceRecognitionError("Similarity threshold must be between -1 and 1")
    query = normalize(embedding)
    if query.size != database.embeddings.shape[1]:
        raise FaceRecognitionError("Query and database embedding dimensions do not match")
    similarities = database.embeddings @ query
    best_index = int(np.argmax(similarities))
    score = float(similarities[best_index])
    known = score >= threshold
    return IdentityMatch(
        label=database.labels[best_index] if known else "Unknown",
        similarity=score,
        known=known,
    )

