"""Small, explicit configuration loader with safe local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional during bootstrap
    load_dotenv = None


@dataclass(frozen=True)
class Settings:
    project_root: Path
    db_path: Path
    evidence_dir: Path
    environment: str
    demo_mode: bool
    federation_online: bool
    dataset_root: Path | None
    sync_endpoint: str | None
    sync_token: str | None


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(project_root: Path | None = None) -> Settings:
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    if load_dotenv is not None:
        load_dotenv(root / ".env", override=False)
    db_value = Path(os.getenv("SAFESIGHT_DB_PATH", "data/safesight.sqlite3"))
    evidence_value = Path(os.getenv("SAFESIGHT_EVIDENCE_DIR", "data/evidence"))
    db_path = db_value if db_value.is_absolute() else root / db_value
    evidence_dir = evidence_value if evidence_value.is_absolute() else root / evidence_value
    dataset_value = os.getenv("SAFESIGHT_DATASET_ROOT", "").strip()
    dataset_root = Path(dataset_value).resolve() if dataset_value else None
    sync_endpoint = os.getenv("SAFESIGHT_SYNC_ENDPOINT", "").strip() or None
    sync_token = os.getenv("SAFESIGHT_SYNC_TOKEN", "").strip() or None
    return Settings(
        project_root=root,
        db_path=db_path,
        evidence_dir=evidence_dir,
        environment=os.getenv("SAFESIGHT_ENV", "development"),
        demo_mode=_as_bool(os.getenv("SAFESIGHT_DEMO_MODE"), True),
        federation_online=_as_bool(os.getenv("SAFESIGHT_FEDERATION_ONLINE"), True),
        dataset_root=dataset_root,
        sync_endpoint=sync_endpoint,
        sync_token=sync_token,
    )
