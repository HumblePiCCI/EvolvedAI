from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from society.analysis import build_lineage_report, render_lineage_report
from society.storage import StorageManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect one lineage and its descendants.")
    parser.add_argument("lineage_id")
    parser.add_argument("--root-dir", default=str(ROOT / "data"))
    parser.add_argument("--db-path", default=str(ROOT / "data/db.sqlite"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    storage = StorageManager(root_dir=Path(args.root_dir), db_path=Path(args.db_path))
    try:
        report = build_lineage_report(storage, args.lineage_id)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_lineage_report(report))
    finally:
        storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
