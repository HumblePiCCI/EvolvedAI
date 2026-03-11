from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from society.storage import StorageManager
from society.trust import compute_drift_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect drift metrics for a generation.")
    parser.add_argument("generation_id", type=int)
    args = parser.parse_args()

    storage = StorageManager(root_dir=ROOT / "data", db_path=ROOT / "data/db.sqlite")
    try:
        artifacts = storage.list_generation_artifacts(args.generation_id)
        memorials = storage.list_generation_memorials(args.generation_id)
        metrics = compute_drift_metrics(
            artifacts,
            memorials,
            communications=storage.list_generation_communications(args.generation_id),
        )
        print(json.dumps(metrics.model_dump(mode="json"), indent=2, sort_keys=True))
    finally:
        storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

