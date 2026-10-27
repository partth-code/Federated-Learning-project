"""Verified Ultralytics PPE adapter; generic COCO weights are rejected."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .contracts import Detection, validate_ppe_labels, normalize_ppe_label


def load_ppe_model(model_path: Path):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    labels = {str(value).lower() for value in model.names.values()}
    validate_ppe_labels(labels)
    return model


def predict(model, source: str | Path, confidence: float = 0.25) -> list[Detection]:
    timestamp = datetime.now(timezone.utc).isoformat()
    results = model.predict(source=source, conf=confidence, verbose=False)
    detections: list[Detection] = []
    for result in results:
        names = result.names
        boxes = result.boxes
        for index in range(len(boxes)):
            class_id = int(boxes.cls[index].item())
            coords = tuple(float(value) for value in boxes.xyxy[index].tolist())
            detections.append(Detection(normalize_ppe_label(names[class_id]), float(boxes.conf[index].item()), coords, timestamp))
    return detections
