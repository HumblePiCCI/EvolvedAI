from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from society.storage import StorageManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect one lineage record.")
    parser.add_argument("lineage_id")
    args = parser.parse_args()

    storage = StorageManager(root_dir=ROOT / "data", db_path=ROOT / "data/db.sqlite")
    try:
        lineage = storage.get_lineage(args.lineage_id)
        if lineage is None:
            raise SystemExit(f"lineage {args.lineage_id} not found")
        print(lineage.model_dump_json(indent=2))
    finally:
        storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

