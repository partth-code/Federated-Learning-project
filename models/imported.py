"""Provenance for user-supplied model bundles, scoped to a registry version."""
from __future__ import annotations
import json
from pathlib import Path


def imported_model_details(project_root: Path, version: dict | None) -> dict | None:
    if not version or not version.get("artifact_path"):
        return None
    root = Path(project_root).resolve()
    artifact = (root / version["artifact_path"]).resolve()
    if not artifact.is_relative_to(root):
        return None
    manifest = artifact.parent / "import_manifest.json"
    if not manifest.is_file():
        return None
    try:
        result = json.loads(manifest.read_text(encoding="utf-8"))
        return result if result.get("model_version_id") == version["version_id"] else None
    except (ValueError, OSError, AttributeError):
        return None
