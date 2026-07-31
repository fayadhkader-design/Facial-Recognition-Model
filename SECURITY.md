# Security and privacy policy

## Sensitive data

Face photos and embeddings are biometric data. Keep them outside the repository,
restrict filesystem access, encrypt backups, and delete them when they are no
longer needed. An embedding is not anonymous simply because it is numeric.

The repository ignores common image formats, model weights, reference folders,
databases, and output folders. Run this audit before every commit:

```bash
python scripts/security_audit.py
```

Installing the development dependencies also provides a local pre-commit hook:

```bash
pre-commit install
pre-commit run --all-files
```

The audit is defense in depth, not a guarantee. Always inspect `git status` and
`git diff --cached` before committing.

## Consent and acceptable use

- Obtain informed consent from every enrolled person.
- Explain where their photos and embeddings are stored and how to request deletion.
- Do not collect images covertly or scrape them from social media.
- Do not use recognition results as proof of identity.
- Do not use this project for surveillance or consequential decisions.

## Reporting a vulnerability

Do not include private photos, embeddings, credentials, or exploit details in a
public issue. Use GitHub's private vulnerability reporting option under the
repository's **Security** tab when available. Otherwise, open a minimal public
issue requesting a private contact channel without disclosing sensitive details.

## Supported versions

This learning project currently supports only the latest commit on `main`.
