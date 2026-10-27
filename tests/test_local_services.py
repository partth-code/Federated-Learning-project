from pathlib import Path
import json
import sqlite3

from analytics.queries import grouped_counts, incident_counts
from database import init_db, insert_incident
from edge.evidence import store_local_evidence
from incidents.schema import Incident
from privacy.status import current_status


def test_local_evidence_is_deduplicated_and_analytics_are_real(tmp_path: Path) -> None:
    db = init_db(tmp_path / "db.sqlite3")
    source = tmp_path / "frame.png"
    source.write_bytes(b"local evidence")
    first = store_local_evidence(source, tmp_path / "evidence", "evt-1")
    second = store_local_evidence(source, tmp_path / "evidence", "evt-1")
    assert first == second
    insert_incident(db, Incident("evt-1", "factory-a", "factory-a-camera-1", "2026-01-01T00:00:00Z", "zone_intrusion"))
    assert incident_counts(db)["total"] == 1
    assert grouped_counts(db, "event_type") == [("zone_intrusion", 1)]
    status = current_status(db, federation_online=False)
    assert status["raw_cctv_federated"] is False
    assert status["federation_online"] is False


def test_privacy_center_shows_only_recorded_secagg_runs(tmp_path: Path) -> None:
    db = init_db(tmp_path / "db.sqlite3")
    assert current_status(db, federation_online=True)["secure_aggregation"].startswith("DISABLED")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO federation_rounds(round_id, round_number, status, metrics_json) VALUES (?, ?, ?, ?)",
            ("test-secagg-r1", 1, "completed", json.dumps({"secure_aggregation": True})),
        )
    assert current_status(db, federation_online=True)["secure_aggregation"].startswith("ENABLED")


def test_privacy_center_reports_measured_ppe_federation_scope(tmp_path: Path) -> None:
    db = init_db(tmp_path / "db.sqlite3")
    evidence = {
        "workload": "Ultralytics YOLO11n Construction-PPE detection-head fine-tuning",
        "simulated_sites": ["factory-a", "factory-b", "factory-c"],
        "site_train_images": {"factory-a": 23, "factory-b": 28, "factory-c": 7},
        "bytes_per_client_update": 1_731_204,
        "raw_training_images_sent": False,
        "secure_aggregation": False,
        "differential_privacy": False,
    }
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO federation_rounds(round_id, round_number, status, metrics_json) VALUES (?, ?, ?, ?)",
            ("ppe-test-r1", 1, "completed", json.dumps(evidence)),
        )

    status = current_status(db, federation_online=True)
    assert status["update_bytes"] == 1_731_204
    assert status["ppe_federation"]["simulated_clients"] == 3
    assert status["ppe_federation"]["train_images_per_client"] == evidence["site_train_images"]
    assert status["ppe_federation"]["raw_training_images_sent"] is False
    assert status["ppe_federation"]["secure_aggregation"] is False
    assert status["ppe_federation"]["differential_privacy"] is False
