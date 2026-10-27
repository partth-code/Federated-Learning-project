from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings
from detector.ultralytics_adapter import load_ppe_model, predict


def main() -> int:
    parser = argparse.ArgumentParser(description="Run verified PPE inference on a local image/video source")
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/runs/construction_ppe_demo/weights/best.pt"))
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    settings = load_settings()
    model_path = args.model if args.model.is_absolute() else settings.project_root / args.model
    model = load_ppe_model(model_path)
    detections = predict(model, args.source, args.confidence)
    print(json.dumps([d.__dict__ for d in detections], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
