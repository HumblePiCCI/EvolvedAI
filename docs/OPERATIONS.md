# OPERATIONS

Bootstrap commands:

```bash
uv sync
uv run pytest
uv run python scripts/run_generation.py --config config/defaults.yaml
uv run python scripts/run_generation.py --config config/defaults.yaml --mode inheritance_off
uv run python scripts/run_experiment.py --config config/defaults.yaml --generations 5 --mode memorials_only
uv run python scripts/run_hypothesis_suite.py --config config/defaults.yaml --generations 5
./scripts/verify_ci.sh
```

The default provider is `mock`, so CI and local tests do not require live API
access.

Artifacts, logs, and summaries are stored in `data/`.

The verification script avoids the heavy training stack in CI and still checks
lint, typing, tests, a generation run, and replay.
