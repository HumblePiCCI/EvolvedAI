from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from society.replay import render_generation_timeline
from society.storage import StorageManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a stored generation as a readable timeline.")
    parser.add_argument("generation_id", type=int)
    parser.add_argument("--root-dir", default=str(ROOT / "data"))
    parser.add_argument("--db-path", default=str(ROOT / "data/db.sqlite"))
    args = parser.parse_args()

    storage = StorageManager(root_dir=Path(args.root_dir), db_path=Path(args.db_path))
    try:
        print(render_generation_timeline(storage, args.generation_id))
    finally:
        storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
