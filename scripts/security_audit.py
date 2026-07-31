#!/usr/bin/env python3
"""Reject commonly leaked biometric artifacts, secrets, and model binaries."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_DIRECTORIES = {"data", "models", "outputs", "references", "results"}
FORBIDDEN_SUFFIXES = {
    ".heic",
    ".jpeg",
    ".jpg",
    ".key",
    ".npy",
    ".npz",
    ".onnx",
    ".pem",
    ".png",
    ".webp",
}
SECRET_PATTERNS = {
    "GitHub token": re.compile(rb"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def tracked_files() -> list[Path]:
    result = subprocess.run(["git", "ls-files", "-z"], check=True, capture_output=True)
    return [Path(item.decode()) for item in result.stdout.split(b"\0") if item]


def audit(paths: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in paths:
        if FORBIDDEN_DIRECTORIES.intersection(path.parts):
            findings.append(f"private artifact directory is tracked: {path}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"private or binary artifact is tracked: {path}")
        try:
            content = path.read_bytes()
        except OSError as exc:
            findings.append(f"cannot inspect tracked file {path}: {exc}")
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                findings.append(f"possible {label} in tracked file: {path}")
    return findings


def main() -> int:
    findings = audit(tracked_files())
    if findings:
        print("Privacy audit failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 1
    print("Privacy audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
