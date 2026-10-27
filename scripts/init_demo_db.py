from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support both `python scripts/init_demo_db.py` and module execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings
from database import init_db, list_sites


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize the local SafeSight-FL demo database")
    parser.add_argument("--reset", action="store_true", help="recreate the local database")
    args = parser.parse_args()
    settings = load_settings()
    path = init_db(settings.db_path, reset=args.reset)
    print(f"Database: {path}")
    for site in list_sites(path):
        print(f"{site['site_id']}: {site['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
