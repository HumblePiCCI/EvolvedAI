from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from society.storage import StorageManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect stored artifacts for a generation.")
    parser.add_argument("generation_id", type=int)
    args = parser.parse_args()

    storage = StorageManager(root_dir=ROOT / "data", db_path=ROOT / "data/db.sqlite")
    try:
        for artifact in storage.list_generation_artifacts(args.generation_id):
            print(artifact.model_dump_json(indent=2))
    finally:
        storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

