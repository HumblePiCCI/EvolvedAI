from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from society.storage import StorageManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay one agent lifespan from JSONL logs.")
    parser.add_argument("generation_id", type=int)
    parser.add_argument("agent_id")
    args = parser.parse_args()

    log_path = ROOT / "data" / "logs" / f"generation_{args.generation_id}" / f"{args.agent_id}.jsonl"
    if not log_path.exists():
        raise SystemExit(f"log file not found: {log_path}")
    print(log_path.read_text(encoding="utf-8"))
    storage = StorageManager(root_dir=ROOT / "data", db_path=ROOT / "data/db.sqlite")
    try:
        evals = [record for record in storage.list_generation_evals(args.generation_id) if record.agent_id == args.agent_id]
        for record in evals:
            print(record.model_dump_json(indent=2))
    finally:
        storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

