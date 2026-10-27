"""Inspect local PPE checkpoint compatibility without training or networking."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings
from detector.ultralytics_adapter import load_ppe_model
from federated.vision_adapter import PPEHeadAdapter


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    settings = load_settings()
    checkpoint = settings.project_root / "artifacts" / "runs" / "construction_ppe_demo" / "weights" / "best.pt"
    if not checkpoint.exists():
        print(json.dumps({"status": "Experiment not run yet", "reason": "local PPE checkpoint is unavailable"}))
        return 0

    model = load_ppe_model(checkpoint)
    adapter = PPEHeadAdapter(model)
    payload = {
        "status": "compatible head-exchange adapter; PPE federation not run",
        "mode": "experimental frozen-backbone detection-head boundary",
        "checkpoint": checkpoint.relative_to(settings.project_root).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "class_count": len(adapter.class_names),
        "trainable_head_parameter_count": adapter.trainable_parameter_count,
        "full_model_parameter_count": sum(parameter.numel() for parameter in model.model.parameters()),
        "bytes_per_client_update": adapter.bytes_per_update,
        "schema_sha256": adapter.schema_sha256,
        "training_rounds": 0,
        "validation_metrics": None,
        "secure_aggregation": False,
        "differential_privacy": False,
    }
    artifact = settings.project_root / "artifacts" / "federated" / "ppe_head_adapter_check.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
