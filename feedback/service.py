"""Human review actions stored locally."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from audit.hash_chain import append_audit


DECISIONS = frozenset({"confirmed", "false_positive", "uncertain"})


def record_feedback(db_path: Path, event_id: str, decision: str, reviewer: str = "local-supervisor") -> str:
    if decision not in DECISIONS:
        raise ValueError(f"Decision must be one of {sorted(DECISIONS)}")
    feedback_id = str(uuid.uuid4())
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO feedback(feedback_id, event_id, decision, reviewer, created_at) VALUES (?, ?, ?, ?, ?)",
            (feedback_id, event_id, decision, reviewer, datetime.now(timezone.utc).isoformat()),
        )
        conn.execute("UPDATE incidents SET feedback = ?, status = 'reviewed' WHERE event_id = ?", (decision, event_id))
    append_audit(db_path, "feedback.recorded", {"event_id": event_id, "decision": decision, "reviewer": reviewer})
    return feedback_id


def feedback_summary(db_path: Path) -> dict[str, int]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT decision, COUNT(*) FROM feedback GROUP BY decision").fetchall()
    return {decision: int(count) for decision, count in rows}
