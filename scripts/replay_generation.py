from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from society.storage import StorageManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a stored generation summary.")
    parser.add_argument("generation_id", type=int)
    args = parser.parse_args()

    storage = StorageManager(root_dir=ROOT / "data", db_path=ROOT / "data/db.sqlite")
    try:
        generation = storage.get_generation(args.generation_id)
        if generation is None:
            raise SystemExit(f"generation {args.generation_id} not found")
        print(json.dumps(generation.model_dump(mode="json"), indent=2, sort_keys=True))
        print(f"agents={len(storage.list_generation_agents(args.generation_id))}")
        print(f"artifacts={len(storage.list_generation_artifacts(args.generation_id))}")
        print(f"evals={len(storage.list_generation_evals(args.generation_id))}")
    finally:
        storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

