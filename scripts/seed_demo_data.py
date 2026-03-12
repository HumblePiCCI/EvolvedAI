from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from society.orchestrator import run_generation_from_config


def main() -> int:
    run_generation_from_config(config_path="config/defaults.yaml", repo_root=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

