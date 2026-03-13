from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from society.orchestrator import run_generation_from_config
from society.constants import EXPERIMENT_MODES


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one AutoCiv generation.")
    parser.add_argument("--config", default="config/defaults.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generation-id", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Reserved for future resume support.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=EXPERIMENT_MODES, default=None)
    args = parser.parse_args()

    summary = run_generation_from_config(
        config_path=args.config,
        repo_root=ROOT,
        generation_id=args.generation_id,
        seed=args.seed,
        dry_run=args.dry_run,
        mode=args.mode,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
