"""Loopback-only SQLite receiver for the one-laptop metadata sync demo."""

from __future__ import annotations

import hmac
import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sync.transport import ALLOWED_EVENT_TYPES, ALLOWED_PAYLOAD_FIELDS, MAX_BODY_BYTES


def create_receiver_server(host: str, port: int, db_path: Path, token: str) -> ThreadingHTTPServer:
    if host.lower() == "localhost":
        host = "127.0.0.1"
    if host != "127.0.0.1":
        raise ValueError("The demo receiver binds only to loopback; use a secured remote receiver for other hosts")
    if not token:
        raise ValueError("SAFESIGHT_SYNC_TOKEN must be configured before starting the receiver")
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS received_incidents (
                queue_id TEXT PRIMARY KEY,
                item_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )

    class Handler(BaseHTTPRequestHandler):
        server_version = "SafeSightLocalReceiver/1"

        def _reply(self, status: int, payload: dict[str, str]) -> None:
            content = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self) -> None:
            if self.path != "/v1/incidents":
                self._reply(404, {"error": "not_found"})
                return
            if not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                self._reply(401, {"error": "unauthorized"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                content_length = -1
            if not 0 < content_length <= MAX_BODY_BYTES:
                self._reply(413 if content_length > MAX_BODY_BYTES else 400, {"error": "invalid_size"})
                return
            try:
                envelope = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if not isinstance(envelope, dict) or set(envelope) != {"schema_version", "queue_id", "item_id", "item_type", "payload"}:
                    raise ValueError
                queue_id = envelope["queue_id"]
                item_id = envelope["item_id"]
                payload = envelope["payload"]
                if (
                    type(envelope["schema_version"]) is not int
                    or envelope["schema_version"] != 1
                    or envelope["item_type"] != "incident_metadata"
                    or not isinstance(queue_id, str)
                    or not queue_id
                    or not isinstance(item_id, str)
                    or not item_id
                    or self.headers.get("Idempotency-Key") != queue_id
                    or not isinstance(payload, dict)
                    or set(payload) != ALLOWED_PAYLOAD_FIELDS
                    or any(not isinstance(payload[field], str) or not payload[field] for field in ALLOWED_PAYLOAD_FIELDS)
                    or payload["event_id"] != item_id
                    or payload["event_type"] not in ALLOWED_EVENT_TYPES
                ):
                    raise ValueError
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, KeyError, TypeError):
                self._reply(400, {"error": "invalid_envelope"})
                return

            serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            with sqlite3.connect(database_path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO received_incidents(queue_id, item_type, payload_json) VALUES (?, ?, ?)",
                    (queue_id, "incident_metadata", serialized),
                )
                existing = conn.execute(
                    "SELECT payload_json FROM received_incidents WHERE queue_id = ?", (queue_id,)
                ).fetchone()
            if existing is None or existing[0] != serialized:
                self._reply(409, {"error": "idempotency_conflict"})
                return
            self._reply(200, {"acknowledged": queue_id})

        def log_message(self, format: str, *args: object) -> None:
            # Avoid logging request data; the demo only needs the HTTP response.
            return

    return ThreadingHTTPServer((host, port), Handler)
