"""Explicit HTTPS/loopback delivery for queued incident metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from audit.hash_chain import append_audit
from edge.offline_queue import mark_attempt, mark_synced, pending


MAX_ITEMS_PER_CALL = 50
MAX_BODY_BYTES = 8192
ALLOWED_EVENT_TYPES = {"ppe_violation", "restricted_zone_intrusion"}
ALLOWED_PAYLOAD_FIELDS = {"event_id", "site_id", "camera_id", "event_type"}


@dataclass(frozen=True)
class SyncResult:
    attempted: int
    synced: int
    failed: int
    unsupported: int
    pending_after: int


def validate_sync_endpoint(endpoint: str, token: str | None) -> None:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("Sync endpoint must be a complete HTTPS URL (or loopback HTTP for the local demo)")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Credentials, query strings, and fragments are not allowed in the sync endpoint")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("Sync endpoint has an invalid port") from exc
    if parsed.scheme == "http" and parsed.hostname.lower() not in {"127.0.0.1", "localhost"}:
        raise ValueError("Unencrypted HTTP is allowed only for a loopback demo receiver")
    if not token:
        raise ValueError("Configure SAFESIGHT_SYNC_TOKEN before enabling queue delivery")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _valid_metadata(item: dict[str, object]) -> dict[str, str] | None:
    if item.get("item_type") != "incident_metadata":
        return None
    try:
        payload = json.loads(str(item["payload_json"]))
    except (KeyError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != ALLOWED_PAYLOAD_FIELDS:
        return None
    if any(not isinstance(payload[field], str) or not payload[field] for field in ALLOWED_PAYLOAD_FIELDS):
        return None
    if payload["event_id"] != item.get("item_id") or payload["event_type"] not in ALLOWED_EVENT_TYPES:
        return None
    return payload


def sync_pending(
    db_path: Path,
    *,
    endpoint: str | None,
    token: str | None,
    online: bool,
    max_items: int = MAX_ITEMS_PER_CALL,
    timeout_seconds: float = 5.0,
) -> SyncResult:
    """Manually send pending incident metadata; never sends image/evidence bytes.

    The sender marks an item synced only after receiving a matching queue-ID
    acknowledgement. Other queue types and malformed payloads remain pending.
    """
    db_path = Path(db_path)
    if not online:
        raise ValueError("Reconnect the demo federation toggle before sending pending metadata")
    if not endpoint:
        raise ValueError("Configure SAFESIGHT_SYNC_ENDPOINT before sending pending metadata")
    if not 1 <= max_items <= MAX_ITEMS_PER_CALL:
        raise ValueError(f"max_items must be between 1 and {MAX_ITEMS_PER_CALL}")
    validate_sync_endpoint(endpoint, token)

    queued = pending(db_path)
    attempted = synced = failed = unsupported = 0
    opener = build_opener(_NoRedirect)
    for item in queued[:max_items]:
        payload = _valid_metadata(item)
        if payload is None:
            unsupported += 1
            continue

        envelope = {
            "schema_version": 1,
            "queue_id": str(item["queue_id"]),
            "item_id": str(item["item_id"]),
            "item_type": "incident_metadata",
            "payload": payload,
        }
        body = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(body) > MAX_BODY_BYTES:
            unsupported += 1
            continue

        mark_attempt(db_path, str(item["queue_id"]))
        attempted += 1
        request = Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                "Idempotency-Key": str(item["queue_id"]),
            },
            method="POST",
        )
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                response_body = response.read(MAX_BODY_BYTES + 1)
                if len(response_body) > MAX_BODY_BYTES:
                    failed += 1
                    continue
            acknowledgment = json.loads(response_body.decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, HTTPException, UnicodeDecodeError, json.JSONDecodeError):
            failed += 1
            continue

        queue_id = str(item["queue_id"])
        if not isinstance(acknowledgment, dict) or acknowledgment.get("acknowledged") != queue_id:
            failed += 1
            continue
        if mark_synced(db_path, queue_id):
            append_audit(db_path, "sync.acknowledged", {"queue_id": queue_id})
            synced += 1
        else:
            failed += 1

    return SyncResult(
        attempted=attempted,
        synced=synced,
        failed=failed,
        unsupported=unsupported,
        pending_after=len(pending(db_path)),
    )
