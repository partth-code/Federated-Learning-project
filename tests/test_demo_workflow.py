"""Synthetic integration fixtures only; never seed the user's demo database."""
import threading

from database import init_db
from detector.contracts import Detection
from edge.incident_pipeline import record_detection_events
from edge.offline_queue import pending
from feedback.service import record_feedback
from ai_copilot.provider import generate_explanation, structured_payload
from analytics.queries import safety_analytics
from audit.hash_chain import verify_chain
from sync.receiver import create_receiver_server
from sync.transport import sync_pending


def test_offline_review_report_analytics_audit_then_sync(tmp_path):
    db = init_db(tmp_path / "local.sqlite3")
    events = record_detection_events(db_path=db,
        detections=[Detection("no_helmet", 0.9, (1, 2, 10, 20), "2026-10-09T10:00:00Z")],
        site_id="factory-a", camera_id="factory-a-camera-1", federation_online=False)
    assert len(events) == 1 and len(pending(db)) == 1
    record_feedback(db, events[0].event_id, "confirmed")
    incident = {**events[0].to_dict(), "feedback": "confirmed"}
    report = generate_explanation(incident, allow_remote=False)
    assert not report.outbound_attempted and "confirmed" in report.text
    counts = safety_analytics(db, "2026-10-09", "2026-10-09")["counts"]
    assert counts["total"] == counts["confirmed"] == counts["ppe_violations"] == 1
    token = "temporary-integration-token"
    server = create_receiver_server("127.0.0.1", 0, tmp_path / "receiver.sqlite3", token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = sync_pending(db, endpoint=f"http://127.0.0.1:{server.server_port}/v1/incidents", token=token, online=True)
        assert result.synced == 1 and not pending(db)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert verify_chain(db)[0]


def test_zone_event_vocabulary_matches_analytics_and_copilot(tmp_path):
    from database import insert_incident
    from incidents.schema import Incident
    db = init_db(tmp_path / "local.sqlite3")
    item = Incident("zone", "factory-a", "cam", "2026-10-09T10:00:00Z", "restricted_zone_intrusion")
    insert_incident(db, item)
    assert safety_analytics(db, "2026-10-09", "2026-10-09")["counts"]["zone_events"] == 1
    assert structured_payload(item.to_dict())["event_type"] == "restricted_zone_intrusion"
