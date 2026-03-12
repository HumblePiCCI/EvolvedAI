from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from society.experiment import run_experiment_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a multi-generation AutoCiv experiment batch.")
    parser.add_argument("--config", default="config/defaults.yaml")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--start-generation-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    report = run_experiment_from_config(
        config_path=args.config,
        repo_root=ROOT,
        generations=args.generations,
        start_generation_id=args.start_generation_id,
        seed=args.seed,
        output_prefix=args.output_prefix,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
