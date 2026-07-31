import hashlib
from pathlib import Path

import pytest

from face_recognition.downloader import ModelSpec, download_model
from face_recognition.errors import ModelError


def local_spec(source: Path, content: bytes) -> ModelSpec:
    return ModelSpec(
        filename="model.onnx",
        url=source.as_uri(),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_download_model_verifies_and_moves_atomically(tmp_path: Path) -> None:
    content = b"safe model fixture"
    source = tmp_path / "source.bin"
    source.write_bytes(content)
    destination = tmp_path / "models" / "model.onnx"

    result = download_model(local_spec(source, content), destination)

    assert result == destination
    assert destination.read_bytes() == content


def test_download_model_rejects_bad_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"unexpected")
    spec = ModelSpec("model.onnx", source.as_uri(), "0" * 64)
    destination = tmp_path / "model.onnx"

    with pytest.raises(ModelError, match="Checksum mismatch"):
        download_model(spec, destination)

    assert not destination.exists()


def test_existing_invalid_model_is_not_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"tampered")
    source = tmp_path / "source.bin"
    source.write_bytes(b"valid")

    with pytest.raises(ModelError, match="Existing model failed"):
        download_model(local_spec(source, b"valid"), destination)
