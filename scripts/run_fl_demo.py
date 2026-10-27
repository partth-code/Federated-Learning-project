from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings
from database import init_db
from federated.lightweight_fl import run_federated


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real lightweight Flower integration test")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    settings = load_settings()
    init_db(settings.db_path)
    print(json.dumps(run_federated(settings.db_path, settings.project_root / "artifacts" / "federated", args.rounds), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
