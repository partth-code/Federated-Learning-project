"""Run the loopback-only SafeSight-FL metadata receiver for a local demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings
from sync.receiver import create_receiver_server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Loopback only; defaults to 127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--db", default="data/sync_receiver.sqlite3", help="Receiver's separate local SQLite file")
    args = parser.parse_args()
    settings = load_settings()
    database = Path(args.db)
    if not database.is_absolute():
        database = settings.project_root / database
    try:
        server = create_receiver_server(args.host, args.port, database, settings.sync_token or "")
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Local metadata receiver listening on http://{args.host}:{server.server_port}/v1/incidents")
    print("Loopback-only demo. Stop with Ctrl+C. No CCTV or image endpoint is enabled.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping local metadata receiver.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
