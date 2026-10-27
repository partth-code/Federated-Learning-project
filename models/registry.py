"""Local model-version registry with a validated PPE active-version pointer."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PPE_MODEL_KINDS = {"ppe_detector_full", "ppe_detector_head_only"}


def _project_root(db_path: Path, project_root: Path | None) -> Path:
    return (project_root or Path(db_path).resolve().parent.parent).resolve()


def _resolve_artifact(project_root: Path, stored_path: str | None) -> Path | None:
    if not stored_path:
        return None
    candidate = Path(stored_path)
    resolved = (candidate if candidate.is_absolute() else project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError:
        return None
    return resolved


def register_version(
    db_path: Path,
    artifact_path: Path,
    *,
    round_number: int | None = None,
    validation_metric: float | None = None,
    notes: str = "",
    model_kind: str = "unspecified",
    clients: Sequence[str] = (),
    project_root: Path | None = None,
) -> str:
    artifact_path = Path(artifact_path).resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    if not model_kind.strip():
        raise ValueError("model_kind must not be empty")
    if any(not isinstance(client, str) or not client.strip() for client in clients):
        raise ValueError("client identifiers must be non-empty strings")
    root = _project_root(db_path, project_root)
    try:
        stored_path = artifact_path.relative_to(root).as_posix()
    except ValueError:
        # Retain legacy registration compatibility; an external artifact cannot
        # be activated because it cannot be resolved under the project root.
        stored_path = artifact_path.name
    version_id = f"model-{uuid.uuid4().hex[:10]}"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """INSERT INTO model_versions(
                version_id, round_number, validation_metric, artifact_path, notes,
                created_at, model_kind, clients_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                version_id,
                round_number,
                validation_metric,
                stored_path,
                notes,
                datetime.now(timezone.utc).isoformat(),
                model_kind,
                json.dumps(list(clients)),
            ),
        )
    return version_id


def list_versions(db_path: Path, *, project_root: Path | None = None) -> list[dict[str, Any]]:
    root = _project_root(db_path, project_root)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute("SELECT * FROM model_versions ORDER BY created_at DESC")]
    versions = []
    for row in rows:
        try:
            row["clients"] = json.loads(row.pop("clients_json", "[]"))
        except json.JSONDecodeError:
            row["clients"] = []
        artifact = _resolve_artifact(root, row.get("artifact_path"))
        row["artifact_available"] = bool(artifact and artifact.is_file())
        row["resolved_artifact"] = artifact
        versions.append(row)
    return versions


def get_active_version(db_path: Path, *, project_root: Path | None = None) -> dict[str, Any] | None:
    root = _project_root(db_path, project_root)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """SELECT state.active_version_id, state.previous_version_id, state.updated_at,
                      version.*
               FROM model_registry_state AS state
               JOIN model_versions AS version ON version.version_id = state.active_version_id
               WHERE state.singleton_id = 1"""
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["clients"] = json.loads(result.pop("clients_json", "[]"))
    result["resolved_artifact"] = _resolve_artifact(root, result.get("artifact_path"))
    return result


def resolve_detection_model(
    db_path: Path,
    fallback_path: Path | None,
    *,
    project_root: Path | None = None,
) -> tuple[Path | None, dict[str, Any] | None]:
    active = get_active_version(db_path, project_root=project_root)
    if active and active.get("resolved_artifact"):
        active_path = Path(active["resolved_artifact"])
        if active_path.is_file():
            return active_path, active
    fallback = Path(fallback_path) if fallback_path is not None else None
    return (fallback if fallback and fallback.is_file() else None), None


def activate_version(
    db_path: Path,
    version_id: str,
    *,
    project_root: Path | None = None,
    validator: Callable[[Path], object] | None = None,
) -> dict[str, Any]:
    root = _project_root(db_path, project_root)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT artifact_path, model_kind FROM model_versions WHERE version_id = ?",
            (version_id,),
        ).fetchone()
    if row is None:
        raise KeyError(f"Unknown model version: {version_id}")
    if row[1] not in PPE_MODEL_KINDS:
        raise ValueError("Only registered PPE detector checkpoints can become the active detection model")
    artifact = _resolve_artifact(root, row[0])
    if artifact is None or not artifact.is_file():
        raise FileNotFoundError("The model artifact is missing or resolves outside the project workspace")
    if validator is None:
        from detector.ultralytics_adapter import load_ppe_model

        load_ppe_model(artifact)
    else:
        validator(artifact)

    timestamp = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        current = conn.execute(
            "SELECT active_version_id FROM model_registry_state WHERE singleton_id = 1"
        ).fetchone()
        if not current or current[0] != version_id:
            previous_version_id = current[0] if current else None
            conn.execute(
                """INSERT INTO model_registry_state(singleton_id, active_version_id, previous_version_id, updated_at)
                   VALUES (1, ?, ?, ?)
                   ON CONFLICT(singleton_id) DO UPDATE SET
                       previous_version_id = excluded.previous_version_id,
                       active_version_id = excluded.active_version_id,
                       updated_at = excluded.updated_at""",
                (version_id, previous_version_id, timestamp),
            )
    active = get_active_version(db_path, project_root=project_root)
    if active is None:
        raise RuntimeError("The activated model version could not be read back")
    return active


def rollback_version(
    db_path: Path,
    *,
    project_root: Path | None = None,
    validator: Callable[[Path], object] | None = None,
) -> dict[str, Any]:
    active = get_active_version(db_path, project_root=project_root)
    if active is None or not active.get("previous_version_id"):
        raise ValueError("No previous model version is available for rollback")
    return activate_version(
        db_path,
        str(active["previous_version_id"]),
        project_root=project_root,
        validator=validator,
    )
