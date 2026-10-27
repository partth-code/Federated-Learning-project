"""Typed incident contract shared by detector, storage, feedback, and copilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Incident:
    event_id: str
    site_id: str
    camera_id: str
    timestamp: str
    event_type: str
    person_id: str | None = None
    confidence: float | None = None
    ppe_status: str | None = None
    zone_id: str | None = None
    zone_entered_at: str | None = None
    zone_duration_seconds: float | None = None
    status: str = "open"
    feedback: str | None = None
    evidence_path: str | None = None
    created_at: str = ""
    previous_zone: str | None = None
    current_zone: str | None = None
    video_timestamp_seconds: float | None = None
    zone_exited_at: str | None = None
    zone_status: str | None = None
    severity: str | None = None
    source_key: str | None = None

    def __post_init__(self) -> None:
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
