# OPERATIONS

Bootstrap commands:

```bash
uv sync
uv run pytest
uv run python scripts/run_generation.py --config config/defaults.yaml
```

The default provider is `mock`, so CI and local tests do not require live API
access.

Artifacts, logs, and summaries are stored in `data/`.

