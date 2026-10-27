"""Import a user-supplied PPE bundle and activate best.pt after inference validation."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    from config import load_settings
    from database import init_db
    from detector.ultralytics_adapter import load_ppe_model, predict
    from detector.contracts import normalize_ppe_label
    from models.registry import register_version, activate_version, get_active_version
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True, help='Local authorized image; inference check only')
    args = parser.parse_args()
    settings = load_settings()
    root = settings.project_root
    names = ('best.pt', 'last.pt', 'best.onnx', 'yolo26n.pt')
    for name in names:
        if not (args.source_dir / name).is_file():
            raise FileNotFoundError(name)
    if not args.sample.is_file():
        raise FileNotFoundError('Local smoke-test image missing')
    hashes = {name: digest(args.source_dir / name) for name in names}
    target = root / 'artifacts' / 'weights' / ('user-' + hashes['best.pt'][:12])
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = target / name
        if path.exists() and digest(path) != hashes[name]:
            raise ValueError('Conflicting bundle artifact; refusing overwrite: ' + name)
        if not path.exists():
            shutil.copy2(args.source_dir / name, path)
        if digest(path) != hashes[name]:
            raise ValueError('Artifact copy hash mismatch')
    model = load_ppe_model(target / 'best.pt')
    if model.task != 'detect':
        raise ValueError('A detection checkpoint is required')
    # No database events are created by this inference check.
    detections = predict(model, args.sample)
    ckpt = model.ckpt or {}
    train_args = ckpt.get('train_args') or {}
    results = ckpt.get('train_results') or {}
    stored_model = train_args.get('model')
    safe_config = {key: train_args.get(key) for key in ('epochs', 'imgsz', 'batch', 'fraction', 'task')}
    safe_config['base_model'] = str(stored_model).replace('\\', '/').rsplit('/', 1)[-1]
    init_db(settings.db_path)
    relative = (target / 'best.pt').relative_to(root).as_posix()
    with sqlite3.connect(settings.db_path) as conn:
        existing = conn.execute('SELECT version_id FROM model_versions WHERE artifact_path = ? AND model_kind = ?',
                                (relative, 'ppe_detector_full')).fetchone()
    version_id = existing[0] if existing else register_version(settings.db_path, target / 'best.pt',
        model_kind='ppe_detector_full', project_root=root,
        notes='User-supplied YOLO26 PPE checkpoint; imported validation metrics only; no local held-out evaluation. Presence classes; no explicit missing-PPE classes.')
    manifest = {
        'model_version_id': version_id, 'imported_at': datetime.now(timezone.utc).isoformat(),
        'source': 'User-supplied trained weights', 'architecture': str(model.model.yaml.get('yaml_file', 'unknown')).replace('\\','/').rsplit('/',1)[-1],
        'classes': model.names, 'canonical_classes': {k:normalize_ppe_label(v) for k,v in model.names.items()},
        'training_config_from_checkpoint': safe_config,
        'recorded_training_rows': len(results.get('epoch', [])) if isinstance(results, dict) else None,
        'checkpoint_metrics': ckpt.get('train_metrics'),
        'metrics_scope': 'Imported checkpoint validation metrics; dataset/split provenance not independently verified. Not local held-out test results.',
        'local_test_metrics': None,
        'missing_ppe_classes': sorted(normalize_ppe_label(v) for v in model.names.values() if normalize_ppe_label(v).startswith('no_')),
        'bundle_sha256': hashes,
        'artifact_roles': {'best.pt':'active inference candidate', 'last.pt':'retained final checkpoint',
                           'best.onnx':'retained export; ONNX runtime not verified', 'yolo26n.pt':'retained base checkpoint; not selected for PPE inference'},
        'inference_check': {'sample_source': 'Existing local Construction-PPE test image (different dataset may apply)',
                            'sample_sha256': digest(args.sample), 'sample_name': args.sample.name,
                            'detections': len(detections), 'labels': sorted({d.label for d in detections}),
                            'purpose': 'Load and inference compatibility only; not an accuracy evaluation'},
        'federation_scope': 'Existing YOLO11 federation/DP results do not apply to this checkpoint',
    }
    manifest_path = target / 'import_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    previous = get_active_version(settings.db_path, project_root=root)
    active = activate_version(settings.db_path, version_id, project_root=root)
    print(json.dumps({'version_id':version_id, 'previous_version': None if previous is None else previous['version_id'],
        'active_artifact':active['artifact_path'], 'manifest':manifest_path.relative_to(root).as_posix(),
        'classes':model.names, 'inference_check':manifest['inference_check']}, indent=2))


if __name__ == '__main__':
    main()
