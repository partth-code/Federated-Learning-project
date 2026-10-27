"""Tamper-evident append-only hash chain; not a blockchain."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _entry_hash(event_type: str, payload_json: str, previous_hash: str | None, created_at: str) -> str:
    material = "|".join((event_type, payload_json, previous_hash or "", created_at)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def append_audit(db_path: Path, event_type: str, payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        previous = conn.execute("SELECT entry_hash FROM audit_logs ORDER BY rowid DESC LIMIT 1").fetchone()
        previous_hash = previous[0] if previous else None
        entry_hash = _entry_hash(event_type, payload_json, previous_hash, created_at)
        log_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO audit_logs(log_id, event_type, payload_json, previous_hash, entry_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (log_id, event_type, payload_json, previous_hash, entry_hash, created_at),
        )
    return log_id


def verify_chain(db_path: Path) -> tuple[bool, str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT event_type, payload_json, previous_hash, entry_hash, created_at FROM audit_logs ORDER BY rowid").fetchall()
    previous_hash = None
    for index, (event_type, payload_json, stored_previous, stored_hash, created_at) in enumerate(rows, start=1):
        if stored_previous != previous_hash:
            return False, f"broken previous hash at entry {index}"
        expected = _entry_hash(event_type, payload_json, stored_previous, created_at)
        if expected != stored_hash:
            return False, f"content hash mismatch at entry {index}"
        previous_hash = stored_hash
    return True, f"verified {len(rows)} entries"
