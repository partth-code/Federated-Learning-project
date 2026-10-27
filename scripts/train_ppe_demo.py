"""Run an explicitly short, real Construction-PPE demo training/evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings


def _plain(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _per_class_metrics(evaluation: object, names: object) -> dict[str, dict[str, float]]:
    """Extract metrics only when Ultralytics exposes actual per-class values."""
    box = getattr(evaluation, "box", None)
    if box is None:
        return {}
    raw_indices = getattr(box, "ap_class_index", [])
    indices = list(raw_indices) if raw_indices is not None else []
    metrics: dict[str, dict[str, float]] = {}
    for position, class_index in enumerate(indices):
        label = str(names.get(int(class_index), class_index)) if isinstance(names, dict) else str(class_index)
        metrics[label] = {
            "precision": float(box.p[position]),
            "recall": float(box.r[position]),
            "mAP50": float(box.ap50[position]),
            "mAP50-95": float(box.ap[position]),
        }
    return metrics


def main() -> int:
    from ultralytics import YOLO

    settings = load_settings()
    data_yaml = settings.project_root / "datasets" / "construction-ppe.yaml"
    base_model = settings.project_root / "yolo11n.pt"
    if not base_model.exists():
        raise FileNotFoundError(base_model)
    run_root = settings.project_root / "artifacts" / "runs"
    model = YOLO(str(base_model))
    train_results = model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=320,
        batch=8,
        device="cpu",
        workers=0,
        fraction=0.02,
        project=str(run_root),
        name="construction_ppe_demo",
        exist_ok=True,
        verbose=False,
    )
    best_path = Path(train_results.save_dir) / "weights" / "best.pt"
    trained_model = YOLO(str(best_path if best_path.exists() else Path(train_results.save_dir) / "weights" / "last.pt"))
    evaluation = trained_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=320,
        batch=8,
        device="cpu",
        workers=0,
        plots=True,
        project=str(run_root),
        name="construction_ppe_demo_test",
        exist_ok=True,
        verbose=False,
    )
    summary = {
        "dataset": "Ultralytics Construction-PPE",
        "dataset_config": str(data_yaml.relative_to(settings.project_root)),
        "mode": "DEMO TRAINING",
        "base_model": "yolo11n.pt",
        "epochs": 1,
        "fraction": 0.02,
        "image_size": 320,
        "device": "cpu",
        "train_run": str(Path(train_results.save_dir).relative_to(settings.project_root)),
        "test_run": str(Path(evaluation.save_dir).relative_to(settings.project_root)),
        "train_metrics": _plain(getattr(train_results, "results_dict", {})),
        "test_metrics": _plain(getattr(evaluation, "results_dict", {})),
        "classes": _plain(getattr(trained_model, "names", {})),
        "per_class_test_metrics": _per_class_metrics(evaluation, getattr(trained_model, "names", {})),
        "best_artifact": str(best_path.relative_to(settings.project_root)) if best_path.exists() else None,
    }
    output_path = settings.project_root / "artifacts" / "ppe_demo_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
