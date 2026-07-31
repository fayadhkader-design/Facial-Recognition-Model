"""Download pinned OpenCV models with integrity verification."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from face_recognition.config import ModelPaths
from face_recognition.errors import ModelError


@dataclass(frozen=True, slots=True)
class ModelSpec:
    filename: str
    url: str
    sha256: str


YUNET = ModelSpec(
    filename="face_detection_yunet_2023mar.onnx",
    url=(
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)
SFACE = ModelSpec(
    filename="face_recognition_sface_2021dec.onnx",
    url=(
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(spec: ModelSpec, destination: Path, *, force: bool = False) -> Path:
    """Download one model atomically and reject unexpected content."""
    if destination.exists() and not force:
        if sha256_file(destination) == spec.sha256:
            return destination
        raise ModelError(f"Existing model failed checksum verification: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(spec.url, timeout=60) as response:
                shutil.copyfileobj(response, output)
        actual = sha256_file(temporary)
        if actual != spec.sha256:
            raise ModelError(
                f"Checksum mismatch for {spec.filename}: expected {spec.sha256}, got {actual}"
            )
        temporary.replace(destination)
        return destination
    except (OSError, urllib.error.URLError) as exc:
        raise ModelError(f"Could not download {spec.filename}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_models(directory: Path, *, force: bool = False) -> ModelPaths:
    paths = ModelPaths.in_directory(directory)
    download_model(YUNET, paths.detector, force=force)
    download_model(SFACE, paths.recognizer, force=force)
    return paths
