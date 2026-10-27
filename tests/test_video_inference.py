from __future__ import annotations

import sys
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np

from database import init_db
from detector.contracts import Detection
from edge.video_inference import (
    VideoFrameResult,
    VideoZoneEntry,
    analyze_local_video,
    record_selected_video_frame,
    record_selected_video_zone_entry,
)


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Box:
    def __len__(self):
        return 1

    @property
    def cls(self):
        return [_Scalar(0)]

    @property
    def conf(self):
        return [_Scalar(0.91)]

    @property
    def xyxy(self):
        return [_Coordinates([1, 2, 10, 12])]


class _Coordinates:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class _Result:
    names = {0: "no_helmet"}
    boxes = _Box()

    @staticmethod
    def plot():
        return np.zeros((12, 16, 3), dtype=np.uint8)


class _Model:
    def __init__(self):
        self.calls = 0

    def predict(self, *, source, conf, verbose):
        assert source.shape == (12, 16, 3)
        assert conf == 0.25 and verbose is False
        self.calls += 1
        return [_Result()]


class _Capture:
    def __init__(self, cv2):
        self.cv2 = cv2
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == self.cv2.CAP_PROP_FPS:
            return 2.0
        if prop == self.cv2.CAP_PROP_FRAME_COUNT:
            return 8
        return 0

    def read(self):
        if self.index >= 8:
            return False, None
        self.index += 1
        return True, np.zeros((12, 16, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class _CV2:
    CAP_PROP_FPS = 1
    CAP_PROP_FRAME_COUNT = 2
    INTER_AREA = 3

    def __init__(self):
        self.capture = _Capture(self)

    def VideoCapture(self, _path):
        return self.capture

    @staticmethod
    def resize(frame, dimensions, interpolation):
        assert interpolation == _CV2.INTER_AREA
        return np.zeros((dimensions[1], dimensions[0], 3), dtype=np.uint8)

    @staticmethod
    def imencode(_suffix, frame):
        return True, np.frombuffer(frame.tobytes(), dtype=np.uint8)

    @staticmethod
    def polylines(_image, _points, _closed, _color, _thickness):
        return None


class _TrackedBox:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def __len__(self):
        return 1

    @property
    def cls(self):
        return [_Scalar(6)]

    @property
    def conf(self):
        return [_Scalar(0.88)]

    @property
    def xyxy(self):
        return [_Coordinates(self.coordinates)]

    @property
    def id(self):
        return [_Scalar(7)]


class _TrackedResult:
    names = {6: "Person"}

    def __init__(self, coordinates):
        self.boxes = _TrackedBox(coordinates)

    @staticmethod
    def plot():
        return np.zeros((12, 16, 3), dtype=np.uint8)


class _TrackingModel:
    names = {6: "Person"}

    def __init__(self, coordinates):
        self.coordinates = iter(coordinates)
        self.calls = 0

    def track(self, *, source, persist, conf, verbose):
        assert source.shape == (12, 16, 3)
        assert persist is True and conf == 0.25 and verbose is False
        self.calls += 1
        return [_TrackedResult(next(self.coordinates))]


def test_video_analysis_samples_frames_locally_and_caps_work(monkeypatch, tmp_path):
    cv2 = _CV2()
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    model = _Model()

    result = analyze_local_video(tmp_path / "clip.mp4", model, max_frames=2)

    assert len(result.frames) == 2
    assert result.frames[0].frame_number == 0
    assert result.frames[1].frame_number == 2
    assert result.frames[1].elapsed_seconds == 1.0
    assert result.duration_seconds == 4.0
    assert result.truncated is True
    assert len(result.frames[0].detections) == 1
    assert result.frames[0].detections[0].label == "no_helmet"
    assert result.frames[0].preview_jpeg
    assert result.frames[0].evidence_jpeg
    assert model.calls == 2
    assert cv2.capture.released is True


def test_zone_tracking_records_only_observed_outside_to_inside_transition(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "cv2", _CV2())
    model = _TrackingModel([
        [0, 0, 2, 2],       # outside
        [6, 4, 10, 8],      # enters
        [6, 4, 10, 8],      # remains inside
        [0, 0, 2, 2],       # exits
    ])
    polygon = [(0.4, 0.3), (0.8, 0.3), (0.8, 0.8), (0.4, 0.8)]

    result = analyze_local_video(tmp_path / "clip.mp4", model, zone_polygon=polygon)

    entries = [(frame.frame_number, entry) for frame in result.frames for entry in frame.zone_entries]
    assert model.calls == 4
    assert len(entries) == 1
    assert entries[0][0] == 2
    assert entries[0][1].zone_id == "restricted-demo-zone"
    assert entries[0][1].detection.track_id == "7"


def test_zone_tracking_does_not_claim_entry_when_first_seen_inside(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "cv2", _CV2())
    model = _TrackingModel([[6, 4, 10, 8]] * 4)

    result = analyze_local_video(
        tmp_path / "clip.mp4",
        model,
        zone_polygon=[(0.4, 0.3), (0.8, 0.3), (0.8, 0.8), (0.4, 0.8)],
    )

    assert not any(frame.zone_entries for frame in result.frames)


def test_selected_video_frame_creates_local_incident_and_metadata_only_queue(tmp_path):
    db_path = init_db(tmp_path / "demo.sqlite3")
    timestamp = datetime.now(timezone.utc).isoformat()
    frame = VideoFrameResult(
        frame_number=30,
        elapsed_seconds=1.0,
        detections=(
            Detection("no_helmet", 0.91, (1, 2, 10, 12), timestamp),
            Detection("helmet", 0.96, (1, 2, 10, 12), timestamp),
        ),
        preview_jpeg=b"annotated-preview",
        evidence_jpeg=b"sampled-frame-evidence",
    )

    incidents = record_selected_video_frame(
        db_path=db_path,
        frame=frame,
        site_id="factory-a",
        camera_id="factory-a-camera-1",
        evidence_dir=tmp_path / "evidence",
        federation_online=False,
    )

    assert len(incidents) == 1
    assert incidents[0].ppe_status == "no_helmet"
    assert incidents[0].evidence_path is not None
    assert incidents[0].evidence_path.endswith(".jpg")
    assert len(list((tmp_path / "evidence").glob("*.jpg"))) == 1
    with sqlite3.connect(db_path) as conn:
        payload = json.loads(conn.execute("SELECT payload_json FROM sync_queue").fetchone()[0])
        assert payload == {
            "event_id": incidents[0].event_id,
            "site_id": "factory-a",
            "camera_id": "factory-a-camera-1",
            "event_type": "ppe_violation",
        }
        assert conn.execute("SELECT COUNT(*) FROM audit_logs WHERE event_type='incident.created'").fetchone()[0] == 1


def test_selected_zone_entry_creates_audited_local_event(tmp_path):
    db_path = init_db(tmp_path / "demo.sqlite3")
    timestamp = datetime.now(timezone.utc).isoformat()
    person = Detection("person", 0.88, (6, 4, 10, 8), timestamp, track_id="7")
    entry = VideoZoneEntry(person, "restricted-demo-zone", timestamp, 1.0)
    frame = VideoFrameResult(2, 1.0, (person,), b"preview", b"evidence")

    incidents = record_selected_video_zone_entry(
        db_path=db_path,
        frame=frame,
        entry=entry,
        site_id="factory-a",
        camera_id="factory-a-camera-1",
        evidence_dir=tmp_path / "evidence",
        federation_online=False,
    )

    assert len(incidents) == 1
    assert incidents[0].event_type == "restricted_zone_intrusion"
    assert incidents[0].person_id == "7"
    assert incidents[0].zone_id == "restricted-demo-zone"
    with sqlite3.connect(db_path) as conn:
        payload = json.loads(conn.execute("SELECT payload_json FROM sync_queue").fetchone()[0])
        assert payload["event_type"] == "restricted_zone_intrusion"
        assert "evidence" not in payload
