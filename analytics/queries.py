"""Truthful analytics queries over local incident and review metadata."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import date, timedelta
from collections import Counter


def incident_counts(db_path: Path) -> dict[str, int]:
    with sqlite3.connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        reviewed = conn.execute("SELECT COUNT(*) FROM incidents WHERE status = 'reviewed'").fetchone()[0]
        confirmed = conn.execute("SELECT COUNT(*) FROM incidents WHERE feedback = 'confirmed'").fetchone()[0]
        false_positive = conn.execute("SELECT COUNT(*) FROM incidents WHERE feedback = 'false_positive'").fetchone()[0]
        uncertain = conn.execute("SELECT COUNT(*) FROM incidents WHERE feedback = 'uncertain'").fetchone()[0]
    return {"total": int(total), "reviewed": int(reviewed), "confirmed": int(confirmed), "false_positive": int(false_positive), "uncertain": int(uncertain)}


def grouped_counts(db_path: Path, field: str) -> list[tuple[str, int]]:
    allowed = {"event_type", "site_id", "camera_id", "status", "feedback"}
    if field not in allowed:
        raise ValueError(f"Unsupported grouping field: {field}")
    with sqlite3.connect(db_path) as conn:
        return [(str(value or "unknown"), int(count)) for value, count in conn.execute(f"SELECT {field}, COUNT(*) FROM incidents GROUP BY {field} ORDER BY COUNT(*) DESC")]


def safety_analytics(db_path: Path, start: str, end: str, *, site_id: str | None = None,
                     camera_id: str | None = None) -> dict:
    """Inclusive UTC incident-date window; review counts use current decisions."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if last < first or (last - first).days > 365:
        raise ValueError("Choose an ordered date range of at most 366 days")
    filters = ["date(timestamp) BETWEEN ? AND ?"]
    values = [first.isoformat(), last.isoformat()]
    for field, value in (("site_id", site_id), ("camera_id", camera_id)):
        if value is not None:
            filters.append(f"{field} = ?")
            values.append(value)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(
            "SELECT date(timestamp) AS day, event_type, site_id, camera_id, feedback FROM incidents WHERE "
            + " AND ".join(filters), values)]
        invalid = conn.execute("SELECT COUNT(*) FROM incidents WHERE date(timestamp) IS NULL").fetchone()[0]
    types = Counter(r["event_type"] for r in rows)
    decisions = Counter(r["feedback"] for r in rows)
    counts = {"total": len(rows), "ppe_violations": types["ppe_violation"],
              "zone_events": types["zone_intrusion"] + types["zone_loitering"] + types["restricted_zone_intrusion"],
              **{key: decisions[key] for key in ("confirmed", "false_positive", "uncertain")}}
    counts["unreviewed_or_other"] = len(rows) - sum(counts[k] for k in ("confirmed", "false_positive", "uncertain"))
    days = Counter(r["day"] for r in rows)
    trend = [{"date": (first + timedelta(days=i)).isoformat(),
              "alerts": days[(first + timedelta(days=i)).isoformat()]}
             for i in range((last - first).days + 1)]
    groups = {}
    for field in ("event_type", "site_id", "camera_id"):
        counter = Counter(str(r[field] or "unknown") for r in rows)
        groups[field] = [{field: key, "alerts": count} for key, count in
                         sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))]
    supported = {key: types[key] for key in ("ppe_violation", "zone_intrusion", "zone_loitering", "restricted_zone_intrusion") if types[key]}
    top = max(supported.values(), default=0)
    return {"counts": counts, "daily": trend, "groups": groups,
            "most_common_alert_types": sorted(k for k, v in supported.items() if v == top),
            "invalid_timestamps_in_database": invalid,
            "scope": "Recorded alerts by UTC incident date; latest human decision, not review-action counts"}
