# Contributing

Thanks for helping improve this consent-based learning project.

## Set up the project

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pre-commit install
```

## Before opening a pull request

```bash
ruff check .
mypy src
pytest --cov=face_recognition
python scripts/security_audit.py
```

Keep commits focused and explain any user-visible behavior changes. Add tests for
new behavior and failure cases.

## Never contribute private data

Do not commit real face photos, embeddings, databases, model binaries,
credentials, or machine-specific configuration. Tests must use generated arrays,
mocked model boundaries, or assets whose licenses and privacy status are clearly
documented. Never bypass `.gitignore` to add a person's photo.

## Design principles

- Process data locally by default.
- Fail closed when data or model integrity is uncertain.
- Label uncertain matches as `Unknown`.
- Keep recognition results advisory and human-reviewable.
- Preserve compatibility with Python 3.11 and newer.

By contributing, you agree that your work is available under the MIT license.
