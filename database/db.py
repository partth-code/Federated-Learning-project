"""SQLite schema for local metadata. Raw video and image bytes are not stored here."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from audit.hash_chain import append_audit


SCHEMA = """
CREATE TABLE IF NOT EXISTS sites (
    site_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    label TEXT NOT NULL,
    is_simulated INTEGER NOT NULL DEFAULT 1 CHECK (is_simulated IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS cameras (
    camera_id TEXT PRIMARY KEY,
    site_id TEXT NOT NULL REFERENCES sites(site_id),
    name TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS incidents (
    event_id TEXT PRIMARY KEY,
    site_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    person_id TEXT,
    confidence REAL,
    ppe_status TEXT,
    zone_id TEXT,
    zone_entered_at TEXT,
    zone_duration_seconds REAL,
    previous_zone TEXT,
    current_zone TEXT,
    video_timestamp_seconds REAL,
    zone_exited_at TEXT,
    zone_status TEXT,
    severity TEXT,
    source_key TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    feedback TEXT,
    evidence_path TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES incidents(event_id),
    decision TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS federation_rounds (
    round_id TEXT PRIMARY KEY,
    round_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    metrics_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS model_versions (
    version_id TEXT PRIMARY KEY,
    round_number INTEGER,
    validation_metric REAL,
    artifact_path TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_kind TEXT NOT NULL DEFAULT 'unspecified',
    clients_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS model_registry_state (
    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
    active_version_id TEXT NOT NULL REFERENCES model_versions(version_id),
    previous_version_id TEXT REFERENCES model_versions(version_id),
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    previous_hash TEXT,
    entry_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS privacy_metrics (
    metric_id TEXT PRIMARY KEY,
    mechanism TEXT NOT NULL,
    epsilon REAL,
    delta REAL,
    clipping_norm REAL,
    noise_multiplier REAL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sync_queue (
    queue_id TEXT PRIMARY KEY,
    item_type TEXT NOT NULL,
    item_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_attempt_at TEXT
);
"""


SEED_SITES = (
    ("factory-a", "Factory A", "SIMULATED FEDERATED SITE", 1),
    ("factory-b", "Factory B", "SIMULATED FEDERATED SITE", 1),
    ("factory-c", "Factory C", "SIMULATED FEDERATED SITE", 1),
)


def init_db(db_path: Path, *, reset: bool = False) -> Path:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if reset and db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(incidents)")}
        migrations = {
            "person_id": "TEXT",
            "ppe_status": "TEXT",
            "zone_id": "TEXT",
            "zone_entered_at": "TEXT",
            "zone_duration_seconds": "REAL",
            "previous_zone": "TEXT",
            "current_zone": "TEXT",
            "video_timestamp_seconds": "REAL",
            "zone_exited_at": "TEXT",
            "zone_status": "TEXT",
            "severity": "TEXT",
            "source_key": "TEXT",
        }
        for column, column_type in migrations.items():
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE incidents ADD COLUMN {column} {column_type}")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_incidents_source_key ON incidents(source_key) WHERE source_key IS NOT NULL")
        version_columns = {row[1] for row in conn.execute("PRAGMA table_info(model_versions)")}
        version_migrations = {
            "model_kind": "TEXT NOT NULL DEFAULT 'unspecified'",
            "clients_json": "TEXT NOT NULL DEFAULT '[]'",
        }
        for column, column_type in version_migrations.items():
            if column not in version_columns:
                conn.execute(f"ALTER TABLE model_versions ADD COLUMN {column} {column_type}")
        legacy_versions = conn.execute(
            "SELECT version_id, artifact_path, notes FROM model_versions WHERE model_kind = 'unspecified'"
        ).fetchall()
        for version_id, artifact_path, notes in legacy_versions:
            note = str(notes or "").lower()
            if "ppe" not in note or "not a ppe" in note or "invalid" in note:
                continue
            kind = "ppe_detector_head_only" if "head" in note else "ppe_detector_full"
            clients_json = "[]"
            if artifact_path:
                run_id = Path(str(artifact_path)).stem
                metric_row = conn.execute(
                    "SELECT metrics_json FROM federation_rounds WHERE round_id LIKE ? AND metrics_json IS NOT NULL ORDER BY created_at DESC LIMIT 1",
                    (f"{run_id}-r%",),
                ).fetchone()
                if metric_row:
                    try:
                        import json

                        metrics = json.loads(metric_row[0])
                        clients_json = json.dumps(metrics.get("simulated_sites", []))
                    except (json.JSONDecodeError, TypeError):
                        pass
            conn.execute(
                "UPDATE model_versions SET model_kind = ?, clients_json = ? WHERE version_id = ?",
                (kind, clients_json, version_id),
            )
        conn.executemany(
            "INSERT OR IGNORE INTO sites(site_id, name, label, is_simulated) VALUES (?, ?, ?, ?)",
            SEED_SITES,
        )
        conn.executemany(
            "INSERT OR IGNORE INTO cameras(camera_id, site_id, name, source_kind) VALUES (?, ?, ?, ?)",
            [(f"{site_id}-camera-1", site_id, "Primary camera", "local_demo") for site_id, *_ in SEED_SITES],
        )
    return db_path


def list_sites(db_path: Path) -> list[dict[str, object]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM sites ORDER BY site_id")]


def insert_incident(db_path: Path, incident: object) -> bool:
    payload = incident.to_dict()
    with sqlite3.connect(db_path) as conn:
        try:
            conn.execute(
                """INSERT INTO incidents(
                event_id, site_id, camera_id, event_type, timestamp, person_id,
                confidence, ppe_status, zone_id, zone_entered_at,
                zone_duration_seconds, status, feedback, evidence_path, created_at,
                previous_zone, current_zone, video_timestamp_seconds, zone_exited_at,
                zone_status, severity, source_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(payload[field] for field in (
                    "event_id", "site_id", "camera_id", "event_type", "timestamp",
                    "person_id", "confidence", "ppe_status", "zone_id", "zone_entered_at",
                    "zone_duration_seconds", "status", "feedback", "evidence_path", "created_at",
                    "previous_zone", "current_zone", "video_timestamp_seconds", "zone_exited_at",
                    "zone_status", "severity", "source_key",
                )),
            )
        except sqlite3.IntegrityError:
            if payload["source_key"] and conn.execute(
                "SELECT 1 FROM incidents WHERE source_key = ?", (payload["source_key"],)
            ).fetchone():
                return False
            raise
    append_audit(db_path, "incident.created", {"event_id": payload["event_id"], "event_type": payload["event_type"], "site_id": payload["site_id"]})
    return True


def count_rows(db_path: Path, table: str) -> int:
    allowed = {"sites", "cameras", "incidents", "feedback", "federation_rounds", "model_versions", "audit_logs", "privacy_metrics", "sync_queue"}
    if table not in allowed:
        raise ValueError(f"Unsupported table: {table}")
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
