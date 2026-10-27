import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from detector.contracts import DetectorUnavailable, normalize_ppe_label, validate_ppe_labels
from detector.ultralytics_adapter import predict
from edge.incident_pipeline import record_detection_events
from database import init_db
from models.imported import imported_model_details


def test_supplied_labels_map_without_creating_absence_classes(tmp_path):
    labels = ['Dust Mask', 'Eye Wear', 'Shield', 'boots', 'glove', 'helmet', 'person', 'vest']
    validate_ppe_labels(set(labels))
    canonical = [normalize_ppe_label(label) for label in labels]
    assert canonical == ['dust_mask', 'goggles', 'shield', 'boots', 'gloves', 'helmet', 'person', 'vest']
    boxes = SimpleNamespace(cls=np.arange(8), conf=np.full(8, .8), xyxy=np.ones((8, 4)))
    class Boxes:
        cls, conf, xyxy = boxes.cls, boxes.conf, boxes.xyxy
        def __len__(self):
            return 8
    model = SimpleNamespace(predict=lambda **kwargs: [SimpleNamespace(names=dict(enumerate(labels)), boxes=Boxes())])
    detections = predict(model, 'unused-test-image')
    assert [d.label for d in detections] == canonical
    db = init_db(tmp_path / 'test.sqlite3')
    assert record_detection_events(db_path=db, detections=detections, site_id='factory-a', camera_id='cam') == []


def test_generic_or_unknown_labels_still_rejected():
    with pytest.raises(DetectorUnavailable):
        validate_ppe_labels({'person'})
    with pytest.raises(ValueError):
        validate_ppe_labels({'helmet', 'car'})


def test_manifest_scoped_to_selected_version_and_workspace(tmp_path):
    bundle = tmp_path / 'artifacts' / 'weights' / 'test'
    bundle.mkdir(parents=True)
    (bundle / 'import_manifest.json').write_text(json.dumps({'model_version_id':'new', 'classes':{'0':'helmet'}}))
    version = {'version_id':'new', 'artifact_path':'artifacts/weights/test/best.pt'}
    assert imported_model_details(tmp_path, version)['model_version_id'] == 'new'
    assert imported_model_details(tmp_path, {**version, 'version_id':'old'}) is None
    assert imported_model_details(tmp_path, {**version, 'artifact_path':'../outside/best.pt'}) is None
    assert imported_model_details(tmp_path, None) is None
