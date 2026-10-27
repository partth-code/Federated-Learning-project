from __future__ import annotations

import json
import socket
import sqlite3
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from audit.hash_chain import verify_chain
from database import init_db
from edge.offline_queue import enqueue
from sync.receiver import create_receiver_server
from sync.transport import sync_pending, validate_sync_endpoint


TOKEN = "local-demo-test-token"


def _queued_incident(db_path: Path) -> str:
    return enqueue(
        db_path,
        "incident_metadata",
        "event-123",
        {
            "event_id": "event-123",
            "site_id": "factory-a",
            "camera_id": "factory-a-camera-1",
            "event_type": "ppe_violation",
        },
    )


def test_authenticated_loopback_sync_acknowledges_and_stores_metadata_only(tmp_path):
    sender_db = init_db(tmp_path / "sender.sqlite3")
    receiver_db = tmp_path / "receiver.sqlite3"
    queue_id = _queued_incident(sender_db)
    server = create_receiver_server("127.0.0.1", 0, receiver_db, TOKEN)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = sync_pending(
            sender_db,
            endpoint=f"http://127.0.0.1:{server.server_port}/v1/incidents",
            token=TOKEN,
            online=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert (result.attempted, result.synced, result.failed, result.unsupported, result.pending_after) == (1, 1, 0, 0, 0)
    with sqlite3.connect(sender_db) as conn:
        row = conn.execute("SELECT status, attempts FROM sync_queue WHERE queue_id = ?", (queue_id,)).fetchone()
        assert row == ("synced", 1)
        assert conn.execute("SELECT COUNT(*) FROM audit_logs WHERE event_type='sync.acknowledged'").fetchone()[0] == 1
    with sqlite3.connect(receiver_db) as conn:
        stored = json.loads(conn.execute("SELECT payload_json FROM received_incidents WHERE queue_id = ?", (queue_id,)).fetchone()[0])
        assert stored == {
            "event_id": "event-123",
            "site_id": "factory-a",
            "camera_id": "factory-a-camera-1",
            "event_type": "ppe_violation",
        }
        assert "evidence_path" not in stored and "image" not in stored
    assert verify_chain(sender_db)[0]


def test_offline_or_unconfigured_transport_never_attempts_delivery(tmp_path):
    db_path = init_db(tmp_path / "sender.sqlite3")
    queue_id = _queued_incident(db_path)

    with pytest.raises(ValueError, match="Reconnect"):
        sync_pending(db_path, endpoint="http://127.0.0.1:9/v1/incidents", token=TOKEN, online=False)
    with pytest.raises(ValueError, match="Configure SAFESIGHT_SYNC_ENDPOINT"):
        sync_pending(db_path, endpoint=None, token=TOKEN, online=True)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT attempts, status FROM sync_queue WHERE queue_id = ?", (queue_id,)).fetchone() == (0, "pending")


def test_failed_delivery_keeps_item_pending_and_records_attempt(tmp_path):
    db_path = init_db(tmp_path / "sender.sqlite3")
    queue_id = _queued_incident(db_path)
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    unused_port = probe.getsockname()[1]
    probe.close()

    result = sync_pending(
        db_path,
        endpoint=f"http://127.0.0.1:{unused_port}/v1/incidents",
        token=TOKEN,
        online=True,
        timeout_seconds=0.2,
    )

    assert result.attempted == 1 and result.failed == 1 and result.synced == 0
    assert result.pending_after == 1
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status, attempts, last_attempt_at FROM sync_queue WHERE queue_id = ?", (queue_id,)).fetchone()
        assert row[0:2] == ("pending", 1)
        assert row[2] is not None


def test_remote_requires_https_and_token_and_rejects_url_credentials():
    with pytest.raises(ValueError, match="loopback"):
        validate_sync_endpoint("http://sync.example.test/v1/incidents", TOKEN)
    with pytest.raises(ValueError, match="TOKEN"):
        validate_sync_endpoint("https://sync.example.test/v1/incidents", None)
    with pytest.raises(ValueError, match="Credentials"):
        validate_sync_endpoint("https://user:secret@sync.example.test/v1/incidents", TOKEN)


def test_non_allowlisted_queue_types_are_never_sent(tmp_path):
    db_path = init_db(tmp_path / "sender.sqlite3")
    queue_id = enqueue(db_path, "feedback", "event-456", {"decision": "confirmed"})

    result = sync_pending(
        db_path,
        endpoint="http://127.0.0.1:8767/v1/incidents",
        token=TOKEN,
        online=True,
    )

    assert result.attempted == 0 and result.unsupported == 1 and result.pending_after == 1
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT status, attempts FROM sync_queue WHERE queue_id = ?", (queue_id,)).fetchone() == ("pending", 0)


def test_receiver_rejects_missing_token(tmp_path):
    server = create_receiver_server("127.0.0.1", 0, tmp_path / "receiver.sqlite3", TOKEN)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = Request(
        f"http://127.0.0.1:{server.server_port}/v1/incidents",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with pytest.raises(HTTPError) as exc:
            urlopen(request, timeout=2)
        assert exc.value.code == 401
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
