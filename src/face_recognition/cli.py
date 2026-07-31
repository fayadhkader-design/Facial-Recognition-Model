"""Command-line interface for local face recognition."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from face_recognition import __version__
from face_recognition.config import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_DETECTION_THRESHOLD,
    ModelPaths,
    default_model_directory,
)
from face_recognition.database import load_database, save_database
from face_recognition.detection import FaceDetector, read_image
from face_recognition.downloader import download_models
from face_recognition.embedding import FaceEmbedder
from face_recognition.enrollment import enroll
from face_recognition.errors import FaceRecognitionError
from face_recognition.recognition import annotate_image, recognize_faces, save_image


def path(value: str) -> Path:
    return Path(value).expanduser()


def add_model_directory(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-directory",
        type=path,
        default=default_model_directory(),
        help="Directory containing the YuNet and SFace ONNX files (default: ./models)",
    )


def add_detection_threshold(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=DEFAULT_DETECTION_THRESHOLD,
        help=(
            "Minimum YuNet face-detection confidence "
            f"(default: {DEFAULT_DETECTION_THRESHOLD})"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face-recognition",
        description="Recognize consenting people in photos without uploading biometric data.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download-models", help="Download checksum-pinned models")
    add_model_directory(download)
    download.add_argument("--force", action="store_true", help="Replace existing model files")

    enroll_command = commands.add_parser("enroll", help="Build a database from reference photos")
    enroll_command.add_argument("--references", type=path, required=True)
    enroll_command.add_argument("--database", type=path, required=True)
    add_model_directory(enroll_command)
    add_detection_threshold(enroll_command)

    recognize = commands.add_parser("recognize", help="Identify faces in a group photo")
    recognize.add_argument("--image", type=path, required=True)
    recognize.add_argument("--database", type=path, required=True)
    recognize.add_argument("--output", type=path, required=True)
    recognize.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_COSINE_THRESHOLD,
        help=f"Minimum cosine similarity (default: {DEFAULT_COSINE_THRESHOLD})",
    )
    add_model_directory(recognize)
    add_detection_threshold(recognize)
    return parser


def build_models(
    directory: Path, detection_threshold: float
) -> tuple[FaceDetector, FaceEmbedder]:
    paths = ModelPaths.in_directory(directory)
    return (
        FaceDetector(paths.detector, confidence_threshold=detection_threshold),
        FaceEmbedder(paths.recognizer),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "download-models":
        paths = download_models(args.model_directory, force=args.force)
        return {
            "status": "ok",
            "models": {"detector": str(paths.detector), "recognizer": str(paths.recognizer)},
        }

    detector, embedder = build_models(args.model_directory, args.detection_threshold)
    if args.command == "enroll":
        database = enroll(args.references, detector, embedder)
        save_database(database, args.database)
        return {
            "status": "ok",
            "database": str(args.database),
            "identities": list(database.labels),
            "identity_count": len(database.labels),
        }

    if args.command == "recognize":
        database = load_database(args.database)
        image = read_image(args.image)
        results = recognize_faces(
            image, database, detector, embedder, threshold=args.threshold
        )
        save_image(annotate_image(image, results), args.output)
        return {
            "status": "ok",
            "input": str(args.image),
            "output": str(args.output),
            "face_count": len(results),
            "faces": [result.as_dict() for result in results],
        }

    raise AssertionError(f"Unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(build_parser().parse_args(argv))
    except FaceRecognitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
