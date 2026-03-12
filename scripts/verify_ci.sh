#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COMMON_PACKAGES=(
  --with "pydantic>=2.11.3"
  --with "pyyaml>=6.0.2"
  --with "networkx>=3.4.2"
)

DEV_PACKAGES=(
  --with "mypy>=1.17.1"
  --with "pytest>=8.4.2"
  --with "ruff>=0.13.0"
)

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/autociv-verify.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

TMP_CONFIG="$TMP_DIR/config.yaml"
GENERATION_ID="${AUTOCIV_VERIFY_GENERATION_ID:-9001}"

uv run --no-project "${COMMON_PACKAGES[@]}" python - "$REPO_ROOT/config/defaults.yaml" "$TMP_CONFIG" "$TMP_DIR" <<'PY'
from pathlib import Path
import sys

import yaml

source_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
tmp_root = Path(sys.argv[3])

config = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
storage_root = (tmp_root / "data").resolve()
config["storage"] = {
    "root_dir": str(storage_root),
    "db_path": str(storage_root / "db.sqlite"),
}
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

uv run --no-project "${DEV_PACKAGES[@]}" ruff check
uv run --no-project "${COMMON_PACKAGES[@]}" "${DEV_PACKAGES[@]}" mypy society worlds evals scripts tests
uv run --no-project "${COMMON_PACKAGES[@]}" "${DEV_PACKAGES[@]}" pytest
uv run --no-project "${COMMON_PACKAGES[@]}" python scripts/run_generation.py --config "$TMP_CONFIG" --generation-id "$GENERATION_ID"
uv run --no-project "${COMMON_PACKAGES[@]}" python scripts/replay_generation.py "$GENERATION_ID" \
  --root-dir "$TMP_DIR/data" \
  --db-path "$TMP_DIR/data/db.sqlite" > "$TMP_DIR/replay_generation.txt"
grep -q "^Generation ${GENERATION_ID}$" "$TMP_DIR/replay_generation.txt"
uv run --no-project "${COMMON_PACKAGES[@]}" python scripts/run_experiment.py \
  --config "$TMP_CONFIG" \
  --generations 2 \
  --start-generation-id "$((GENERATION_ID + 100))" > "$TMP_DIR/experiment_report.json"
grep -q "\"generation_ids\"" "$TMP_DIR/experiment_report.json"
