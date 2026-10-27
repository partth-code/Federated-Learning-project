from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from database import init_db
from detector.contracts import Detection
from detector.geometry import ZoneTracker
from edge.incident_pipeline import record_detection_events, record_zone_observations


def test_ppe_events_are_audited_and_offline_queue_contains_metadata_only(tmp_path):
    db = init_db(tmp_path / "demo.sqlite3")
    image = tmp_path / "evidence.jpg"
    image.write_bytes(b"local-image-bytes")
    detections = [
        Detection("no_helmet", 0.91, (1, 2, 10, 20), datetime.now(timezone.utc).isoformat()),
        Detection("helmet", 0.98, (1, 2, 10, 20), datetime.now(timezone.utc).isoformat()),
    ]
    events = record_detection_events(
        db_path=db,
        detections=detections,
        site_id="factory-a",
        camera_id="factory-a-camera-1",
        evidence_source=image,
        evidence_dir=tmp_path / "evidence",
        federation_online=False,
    )
    assert len(events) == 1
    assert events[0].ppe_status == "no_helmet"
    assert list((tmp_path / "evidence").glob("*.jpg"))
    with sqlite3.connect(db) as conn:
        queued = json.loads(conn.execute("SELECT payload_json FROM sync_queue").fetchone()[0])
        assert queued["event_id"] == events[0].event_id
        assert "evidence" not in queued and "image" not in queued
        assert conn.execute("SELECT COUNT(*) FROM audit_logs WHERE event_type='incident.created'").fetchone()[0] == 1


def test_zone_requires_track_id_and_only_records_entry(tmp_path):
    db = init_db(tmp_path / "demo.sqlite3")
    tracker = ZoneTracker("restricted-1", [(0.4, 0.4), (0.8, 0.4), (0.8, 0.9), (0.4, 0.9)])
    now = datetime.now(timezone.utc).isoformat()
    untracked = Detection("person", 0.9, (50, 40, 70, 80), now)
    assert not record_zone_observations(
        db_path=db, detections=[untracked], tracker=tracker, image_size=(100, 100),
        site_id="factory-a", camera_id="factory-a-camera-1",
    )
    tracked = Detection("person", 0.9, (50, 40, 70, 80), now, track_id="track-1")
    events = record_zone_observations(
        db_path=db, detections=[tracked], tracker=tracker, image_size=(100, 100),
        site_id="factory-a", camera_id="factory-a-camera-1",
    )
    assert len(events) == 1
    assert events[0].event_type == "restricted_zone_intrusion"
    assert events[0].zone_id == "restricted-1"
