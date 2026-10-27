"""User-approved full-split GPU training; preserves the existing demo model."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from config import load_settings
    from database import init_db
    from models.registry import register_version
    from scripts.train_ppe_demo import _plain, _per_class_metrics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=512)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
    settings = load_settings()
    root = settings.project_root
    data = root / 'datasets' / 'construction-ppe.yaml'
    check_det_dataset(str(data), autodownload=False)
    run_name = 'construction_ppe_gpu_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    status_path = root / 'artifacts' / 'gpu_training_status.json'
    status = {'status': 'starting', 'run_name': run_name, 'device': torch.cuda.get_device_name(0),
              'epochs_requested': args.epochs, 'batch': args.batch, 'image_size': args.imgsz,
              'fraction': 1.0, 'mode': 'FULL TRAINING', 'torch': torch.__version__}

    def save_status():
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temporary.replace(status_path)

    def progress(trainer):
        status.update(status='training', epochs_completed=trainer.epoch + 1,
                      validation_metrics=_plain(trainer.metrics))
        save_status()

    save_status()
    try:
        model = YOLO(str(root / 'yolo11n.pt'))
        model.add_callback('on_fit_epoch_end', progress)
        trained = model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, device=0, workers=0, fraction=1.0, amp=True,
            patience=12, cache=False, project=str(root / 'artifacts' / 'runs'),
            name=run_name, exist_ok=False, verbose=False, plots=True)
        checkpoint = Path(trained.save_dir) / 'weights' / 'best.pt'
        status.update(status='evaluating', best_artifact=checkpoint.relative_to(root).as_posix())
        save_status()
        best = YOLO(str(checkpoint))
        evaluated = best.val(data=str(data), split='test', device=0, batch=args.batch,
            imgsz=args.imgsz, workers=0, plots=True, project=str(root / 'artifacts' / 'runs'),
            name=run_name + '_test', exist_ok=False, verbose=False)
        init_db(settings.db_path)
        version = register_version(settings.db_path, checkpoint, project_root=root,
            model_kind='ppe_detector_full', validation_metric=float(evaluated.box.map50),
            notes='User-approved full-split GPU training; held-out test mAP50; not activated')
        status.update(status='completed', test_metrics=_plain(evaluated.results_dict),
            per_class_test_metrics=_per_class_metrics(evaluated, best.names),
            model_version_id=version, activated=False)
        save_status()
    except Exception as exc:
        status.update(status='failed', error_type=type(exc).__name__)
        save_status()
        raise


if __name__ == '__main__':
    main()
