import json
from pathlib import Path

import cv2 as cv
import numpy as np

from face_recognition import cli
from face_recognition.database import FaceDatabase, save_database
from face_recognition.detection import DetectedFace


class WorkflowDetector:
    def detect(self, image):
        return [DetectedFace(4, 4, 12, 12, 0.99, np.zeros(15, dtype=np.float32))]


class WorkflowEmbedder:
    def embed(self, image, face):
        return np.array([1.0, 0.0], dtype=np.float32)


def test_recognize_command_end_to_end_with_model_boundary_mocked(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    database_path = tmp_path / "faces.npz"
    save_database(
        FaceDatabase(("Alice",), np.array([[1.0, 0.0]], dtype=np.float32)), database_path
    )
    input_path = tmp_path / "group.png"
    output_path = tmp_path / "result.png"
    assert cv.imwrite(str(input_path), np.zeros((30, 30, 3), dtype=np.uint8))
    monkeypatch.setattr(
        cli,
        "build_models",
        lambda directory, detection_threshold: (WorkflowDetector(), WorkflowEmbedder()),
    )

    exit_code = cli.main(
        [
            "recognize",
            "--image",
            str(input_path),
            "--database",
            str(database_path),
            "--output",
            str(output_path),
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["face_count"] == 1
    assert summary["faces"][0]["label"] == "Alice"
    assert output_path.is_file()
    assert np.any(cv.imread(str(output_path)) != 0)
