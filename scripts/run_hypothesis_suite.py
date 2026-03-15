from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from society.constants import EXPERIMENT_MODES
from society.experiment import run_hypothesis_suite_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a multi-mode AutoCiv hypothesis suite.")
    parser.add_argument("--config", default="config/defaults.yaml")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--start-generation-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--modes", nargs="*", choices=EXPERIMENT_MODES, default=list(EXPERIMENT_MODES))
    parser.add_argument("--provider-name", default=None)
    parser.add_argument("--provider-model", default=None)
    parser.add_argument("--provider-base-url", default=None)
    parser.add_argument("--provider-timeout-seconds", type=float, default=None)
    parser.add_argument("--provider-reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None)
    parser.add_argument("--provider-max-output-tokens", type=int, default=None)
    args = parser.parse_args()

    report = run_hypothesis_suite_from_config(
        config_path=args.config,
        repo_root=ROOT,
        generations=args.generations,
        modes=args.modes,
        start_generation_id=args.start_generation_id,
        seed=args.seed,
        output_prefix=args.output_prefix,
        provider_name=args.provider_name,
        provider_model=args.provider_model,
        provider_base_url=args.provider_base_url,
        provider_timeout_seconds=args.provider_timeout_seconds,
        provider_reasoning_effort=args.provider_reasoning_effort,
        provider_max_output_tokens=args.provider_max_output_tokens,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
