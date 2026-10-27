"""Deterministic structured-event fallback for the optional LLM copilot."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
import sqlite3
from pathlib import Path


def incident_report(incident: dict[str, object]) -> str:
    event_type = incident.get("event_type", "event")
    site_id = incident.get("site_id", "unknown site")
    confidence = incident.get("confidence")
    confidence_text = f" with recorded confidence {confidence:.3f}" if isinstance(confidence, (int, float)) else ""
    feedback = incident.get("feedback") or "not reviewed"
    return f"Structured CV event: {event_type} at {site_id}{confidence_text}. Human feedback: {feedback}. This is a recorded alert, not proof of a violation. Review the evidence and follow site policy."


def supervisor_actions(incident: dict[str, object]) -> list[str]:
    actions = ["Review the local evidence snapshot", "Record Confirm, False Positive, or Uncertain feedback"]
    if incident.get("event_type") in {"ppe_violation", "zone_intrusion", "zone_loitering", "restricted_zone_intrusion"}:
        actions.append("Follow the site's established safety escalation procedure")
    return actions


def daily_summary(incidents: Iterable[dict[str, object]]) -> str:
    items = list(incidents)
    if not items:
        return "No structured incidents were recorded for this period."
    counts: dict[str, int] = {}
    for incident in items:
        event_type = str(incident.get("event_type", "unknown"))
        counts[event_type] = counts.get(event_type, 0) + 1
    details = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
    return f"Recorded structured incidents: {details}. This summary does not determine whether an incident occurred."


def incidents_for_day(db_path: Path, day: str) -> list[dict[str, object]]:
    """All events on the selected UTC calendar date, independent of UI pagination."""
    date.fromisoformat(day)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(
            "SELECT * FROM incidents WHERE date(timestamp) = ? ORDER BY timestamp, event_id", (day,))]
