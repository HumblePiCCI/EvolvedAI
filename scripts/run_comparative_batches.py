from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from society.constants import EXPERIMENT_MODES
from society.experiment import run_comparative_batches_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run matched-seed comparative AutoCiv batches.")
    parser.add_argument("--config", default="config/defaults.yaml")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--start-generation-id", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--modes", nargs="*", choices=EXPERIMENT_MODES, default=list(EXPERIMENT_MODES))
    args = parser.parse_args()

    report = run_comparative_batches_from_config(
        config_path=args.config,
        repo_root=ROOT,
        generations=args.generations,
        seeds=args.seeds,
        modes=args.modes,
        start_generation_id=args.start_generation_id,
        output_prefix=args.output_prefix,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
