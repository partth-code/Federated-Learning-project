from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

import pytest

from audit.hash_chain import append_audit, verify_chain
from database import init_db
from detector.geometry import ZoneTracker, parse_normalized_polygon, point_in_polygon
from edge.offline_queue import enqueue, pending
from feedback.service import record_feedback
from incidents.schema import Incident
from models.registry import (
    activate_version,
    get_active_version,
    list_versions,
    register_version,
    resolve_detection_model,
    rollback_version,
)


def test_zone_entry_and_loitering() -> None:
    assert point_in_polygon((5, 5), [(0, 0), (10, 0), (10, 10), (0, 10)])
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tracker = ZoneTracker("restricted-1", [(0, 0), (10, 0), (10, 10), (0, 10)], loitering_seconds=5)
    assert tracker.update("p1", (5, 5), start)["entered"]
    assert tracker.update("p1", (5, 5), start + timedelta(seconds=6))["loitering"]


def test_parse_normalized_zone_polygon_validates_coordinates() -> None:
    assert parse_normalized_polygon("0.1,0.1;0.9,0.1;0.9,0.9;0.1,0.9") == [
        (0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9),
    ]
    with pytest.raises(ValueError):
        parse_normalized_polygon("0,0;1.2,0;1,1")


def test_audit_chain_detects_tampering(tmp_path: Path) -> None:
    db = init_db(tmp_path / "db.sqlite3")
    append_audit(db, "incident.created", {"event_id": "evt-1"})
    append_audit(db, "feedback.recorded", {"event_id": "evt-1", "decision": "confirmed"})
    assert verify_chain(db)[0]
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE audit_logs SET payload_json = '{}' WHERE event_type = 'feedback.recorded'")
    assert not verify_chain(db)[0]


def test_offline_queue_and_local_feedback(tmp_path: Path) -> None:
    db = init_db(tmp_path / "db.sqlite3")
    event = Incident("evt-1", "factory-a", "factory-a-camera-1", "2026-01-01T00:00:00Z", "zone_intrusion")
    from database.db import insert_incident
    insert_incident(db, event)
    record_feedback(db, "evt-1", "uncertain")
    queue_id = enqueue(db, "feedback", "evt-1", {"decision": "uncertain"})
    assert queue_id
    assert pending(db)[0]["item_id"] == "evt-1"


def test_registry_requires_real_artifact(tmp_path: Path) -> None:
    db = init_db(tmp_path / "db.sqlite3")
    with pytest.raises(FileNotFoundError):
        register_version(db, tmp_path / "missing.pt")
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"test artifact")
    assert register_version(db, artifact, round_number=1)


def test_model_registry_switches_and_rolls_back_only_available_ppe_versions(tmp_path: Path) -> None:
    project_root = tmp_path
    db = init_db(project_root / "data" / "safesight.sqlite3")
    first_artifact = project_root / "artifacts" / "first.pt"
    second_artifact = project_root / "artifacts" / "second.pt"
    first_artifact.parent.mkdir(parents=True)
    first_artifact.write_bytes(b"first valid checkpoint")
    second_artifact.write_bytes(b"second valid checkpoint")
    first = register_version(
        db,
        first_artifact,
        model_kind="ppe_detector_full",
        clients=("factory-a", "factory-b", "factory-c"),
        project_root=project_root,
    )
    second = register_version(
        db,
        second_artifact,
        round_number=1,
        model_kind="ppe_detector_head_only",
        clients=("factory-a", "factory-b", "factory-c"),
        project_root=project_root,
    )
    validator = lambda path: path.read_bytes()

    assert get_active_version(db, project_root=project_root) is None
    assert activate_version(db, first, project_root=project_root, validator=validator)["version_id"] == first
    active = activate_version(db, second, project_root=project_root, validator=validator)
    assert active["version_id"] == second
    assert active["previous_version_id"] == first
    assert active["clients"] == ["factory-a", "factory-b", "factory-c"]
    resolved, resolved_version = resolve_detection_model(
        db,
        project_root / "artifacts" / "unused.pt",
        project_root=project_root,
    )
    assert resolved == second_artifact.resolve()
    assert resolved_version["version_id"] == second
    assert rollback_version(db, project_root=project_root, validator=validator)["version_id"] == first
    assert rollback_version(db, project_root=project_root, validator=validator)["version_id"] == second
    assert len(list_versions(db, project_root=project_root)) == 2


def test_model_registry_rejects_missing_or_non_ppe_activation(tmp_path: Path) -> None:
    project_root = tmp_path
    db = init_db(project_root / "data" / "safesight.sqlite3")
    artifact = project_root / "artifacts" / "classifier.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"toy classifier")
    version_id = register_version(
        db,
        artifact,
        model_kind="toy_classifier",
        project_root=project_root,
    )
    with pytest.raises(ValueError, match="PPE detector"):
        activate_version(db, version_id, project_root=project_root)
    ppe_id = register_version(
        db,
        artifact,
        model_kind="ppe_detector_full",
        project_root=project_root,
    )
    artifact.unlink()
    with pytest.raises(FileNotFoundError, match="missing or resolves outside"):
        activate_version(db, ppe_id, project_root=project_root, validator=lambda path: None)


def test_model_registry_migrates_legacy_ppe_metadata_without_reclassifying_toy_versions(tmp_path: Path) -> None:
    db = tmp_path / "data" / "legacy.sqlite3"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """CREATE TABLE model_versions (
                   version_id TEXT PRIMARY KEY, round_number INTEGER, validation_metric REAL,
                   artifact_path TEXT, notes TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
               );
               CREATE TABLE federation_rounds (
                   round_id TEXT PRIMARY KEY, round_number INTEGER NOT NULL, status TEXT NOT NULL,
                   metrics_json TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
               );"""
        )
        conn.execute(
            "INSERT INTO model_versions(version_id, artifact_path, notes) VALUES (?, ?, ?)",
            ("ppe-v1", "artifacts/federated/ppe-fl-v1.pt", "One-round PPE head demo"),
        )
        conn.execute(
            "INSERT INTO model_versions(version_id, artifact_path, notes) VALUES (?, ?, ?)",
            ("toy-v1", "artifacts/federated/toy.pt", "Flower classifier; not a PPE detector"),
        )
        conn.execute(
            "INSERT INTO federation_rounds(round_id, round_number, status, metrics_json) VALUES (?, ?, ?, ?)",
            (
                "ppe-fl-v1-r1",
                1,
                "completed",
                json.dumps({"simulated_sites": ["factory-a", "factory-b", "factory-c"]}),
            ),
        )

    init_db(db)
    with sqlite3.connect(db) as conn:
        migrated = conn.execute(
            "SELECT model_kind, clients_json FROM model_versions WHERE version_id = ?", ("ppe-v1",)
        ).fetchone()
        toy = conn.execute("SELECT model_kind FROM model_versions WHERE version_id = ?", ("toy-v1",)).fetchone()
        state_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'model_registry_state'"
        ).fetchone()
    assert migrated == ("ppe_detector_head_only", '["factory-a", "factory-b", "factory-c"]')
    assert toy == ("unspecified",)
    assert state_table is not None
