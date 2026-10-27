"""Local pending-work queue used when federation is unavailable."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path


def enqueue(db_path: Path, item_type: str, item_id: str, payload: dict[str, object]) -> str:
    queue_id = str(uuid.uuid4())
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sync_queue(queue_id, item_type, item_id, payload_json) VALUES (?, ?, ?, ?)",
            (queue_id, item_type, item_id, json.dumps(payload, sort_keys=True)),
        )
    return queue_id


def pending(db_path: Path) -> list[dict[str, object]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM sync_queue WHERE status = 'pending' ORDER BY created_at")]


def mark_synced(db_path: Path, queue_id: str) -> bool:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "UPDATE sync_queue SET status = 'synced' WHERE queue_id = ? AND status = 'pending'",
            (queue_id,),
        )
        return cursor.rowcount == 1


def mark_attempt(db_path: Path, queue_id: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE sync_queue SET attempts = attempts + 1, last_attempt_at = CURRENT_TIMESTAMP "
            "WHERE queue_id = ? AND status = 'pending'",
            (queue_id,),
        )
