"""Local-only evidence path policy and duplicate detection."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


ALLOWED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def store_local_evidence(source: Path, evidence_dir: Path, event_id: str) -> Path:
    source = Path(source)
    if source.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("Only local image evidence is supported")
    if not source.exists():
        raise FileNotFoundError(source)
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    destination = evidence_dir / f"{event_id}-{digest}{source.suffix.lower()}"
    if not destination.exists():
        shutil.copy2(source, destination)
    return destination
