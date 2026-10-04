# Agent instructions (Language-engineer)

This repository holds product and domain work for `projects/assessment/apple/siri-language-engineer/language-engineer`.
Domain and stack rules live here. Shared delivery tooling on the machine is out of scope for this file.

## Stack

- Python (pyproject/uv),Conda env file
- Prefer commands and layout already documented in `README.md` and manifests in this repo.

## Conventions

- Detect existing patterns before inventing new ones.
- Keep secrets in environment variables or ignored local files; never commit credentials.
- Prefer small, reviewable changes on a feature branch; do not force-push shared default branches.
- Comments explain why, not what. Plain language; no decorative symbols in commits.

## Security

- Do not log tokens, API keys, or personal data.
- Redact secrets at boundaries (logs, errors, UI).
