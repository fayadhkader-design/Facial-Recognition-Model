from pathlib import Path

from face_recognition.config import MODEL_DIRECTORY_ENV, ModelPaths, default_model_directory


def test_model_paths_use_given_directory(tmp_path: Path) -> None:
    paths = ModelPaths.in_directory(tmp_path)

    assert paths.detector.parent == tmp_path
    assert paths.recognizer.parent == tmp_path


def test_model_directory_can_be_overridden(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(MODEL_DIRECTORY_ENV, str(tmp_path))

    assert default_model_directory() == tmp_path
