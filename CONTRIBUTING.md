# Contributing

Thanks for your interest in contributing to ReTracker!

Please follow our community standards in `CODE_OF_CONDUCT.md`.

## Development setup

```bash
git clone <repo>
cd re-tracker

# Create/activate your environment (example: conda)
conda env create -f environment.yaml
conda activate retracker_env

# Editable install + dev tools
python -m pip install -e ".[dev]"
```

Optional:

```bash
# Optional: initialize git submodules (third-party integrations)
git submodule update --init --recursive

# Runnable apps (streaming UI / extra IO backends)
python -m pip install -e ".[apps]"

# Training / evaluation dependencies (source checkout only)
python -m pip install -e ".[train]"
```

## Code style

We use Ruff for linting and formatting.

```bash
ruff format retracker/cli tests
ruff check retracker/cli tests
```

## Tests

```bash
pytest tests/ -q
```

## Pull requests

- Keep changes focused and easy to review.
- Add/adjust tests for behavior changes.
- Update documentation when you change user-facing CLI or scripts.
- Prefer backwards-compatible changes; if deprecating behavior, add clear warnings.
