"""Turn real detector outputs into local incidents without inventing events."""

from __future__ import annotations

import uuid
from pathlib import Path

from database import insert_incident
from detector.contracts import Detection
from detector.geometry import ZoneTracker
from edge.evidence import store_local_evidence
from edge.offline_queue import enqueue
from incidents.schema import Incident


PPE_VIOLATIONS = frozenset({"no_helmet", "no_vest", "no_gloves", "no_boots", "no_goggle"})


def record_detection_events(
    *,
    db_path: Path,
    detections: list[Detection],
    site_id: str,
    camera_id: str,
    evidence_source: Path | None = None,
    evidence_dir: Path | None = None,
    federation_online: bool = True,
) -> list[Incident]:
    """Persist only supported violation labels returned by the configured PPE model."""
    incidents: list[Incident] = []
    for detection in detections:
        label = detection.label.strip().lower()
        if label not in PPE_VIOLATIONS:
            continue
        event_id = str(uuid.uuid4())
        evidence_path = None
        if evidence_source is not None and evidence_dir is not None:
            evidence_path = str(store_local_evidence(evidence_source, evidence_dir, event_id))
        incident = Incident(
            event_id=event_id,
            site_id=site_id,
            camera_id=camera_id,
            timestamp=detection.timestamp,
            event_type="ppe_violation",
            person_id=detection.track_id,
            confidence=detection.confidence,
            ppe_status=label,
            evidence_path=evidence_path,
        )
        insert_incident(db_path, incident)
        if not federation_online:
            enqueue(db_path, "incident_metadata", event_id, {
                "event_id": event_id,
                "site_id": site_id,
                "camera_id": camera_id,
                "event_type": incident.event_type,
            })
        incidents.append(incident)
    return incidents


def record_zone_observations(
    *,
    db_path: Path,
    detections: list[Detection],
    tracker: ZoneTracker,
    image_size: tuple[int, int],
    site_id: str,
    camera_id: str,
    federation_online: bool = True,
) -> list[Incident]:
    """Record zone entries only for tracked person detections and a configured polygon.

    Polygon coordinates are normalized to [0, 1]. A detection without a tracker ID
    is deliberately ignored; a single still image cannot establish identity/entry.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    incidents: list[Incident] = []
    for detection in detections:
        if detection.label.strip().lower() != "person" or not detection.track_id:
            continue
        x1, _, x2, y2 = detection.bbox_xyxy
        observation = tracker.update(
            detection.track_id,
            ((x1 + x2) / (2 * width), y2 / height),
        )
        if not observation["entered"]:
            continue
        incident = record_zone_entry(
            db_path=db_path,
            detection=detection,
            site_id=site_id,
            camera_id=camera_id,
            zone_id=str(observation["zone_id"]),
            entered_at=str(observation["entered_at"]),
            duration_seconds=float(observation["duration_seconds"]),
            federation_online=federation_online,
        )
        if incident is not None:
            incidents.append(incident)
    return incidents


def record_zone_entry(
    *,
    db_path: Path,
    detection: Detection,
    site_id: str,
    camera_id: str,
    zone_id: str,
    entered_at: str,
    duration_seconds: float = 0.0,
    evidence_source: Path | None = None,
    evidence_dir: Path | None = None,
    federation_online: bool = True,
) -> Incident | None:
    """Persist one actual tracked-person entry observation and its local evidence."""
    if detection.label.strip().lower() != "person" or not detection.track_id:
        return None
    event_id = str(uuid.uuid4())
    evidence_path = None
    if evidence_source is not None and evidence_dir is not None:
        evidence_path = str(store_local_evidence(evidence_source, evidence_dir, event_id))
    incident = Incident(
        event_id=event_id,
        site_id=site_id,
        camera_id=camera_id,
        timestamp=detection.timestamp,
        event_type="restricted_zone_intrusion",
        person_id=detection.track_id,
        confidence=detection.confidence,
        zone_id=zone_id,
        zone_entered_at=entered_at,
        zone_duration_seconds=duration_seconds,
        evidence_path=evidence_path,
    )
    insert_incident(db_path, incident)
    if not federation_online:
        enqueue(db_path, "incident_metadata", event_id, {
            "event_id": event_id,
            "site_id": site_id,
            "camera_id": camera_id,
            "event_type": incident.event_type,
        })
    return incident
