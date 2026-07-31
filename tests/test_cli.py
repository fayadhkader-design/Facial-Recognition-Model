import json
from pathlib import Path

from face_recognition import cli
from face_recognition.config import ModelPaths


def test_download_models_command_emits_json(monkeypatch, tmp_path: Path, capsys) -> None:
    paths = ModelPaths.in_directory(tmp_path)
    monkeypatch.setattr(cli, "download_models", lambda directory, force: paths)

    exit_code = cli.main(["download-models", "--model-directory", str(tmp_path)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["status"] == "ok"
    assert output["models"]["detector"] == str(paths.detector)


def test_enroll_reports_missing_models_without_traceback(tmp_path: Path, capsys) -> None:
    exit_code = cli.main(
        [
            "enroll",
            "--references",
            str(tmp_path / "references"),
            "--database",
            str(tmp_path / "faces.npz"),
            "--model-directory",
            str(tmp_path / "models"),
        ]
    )

    assert exit_code == 2
    assert "error: Face detector model not found" in capsys.readouterr().err


def test_parser_uses_documented_threshold() -> None:
    args = cli.build_parser().parse_args(
        ["recognize", "--image", "group.jpg", "--database", "faces.npz", "--output", "out.jpg"]
    )

    assert args.threshold == 0.363
