from types import SimpleNamespace

import numpy as np

from edge.webcam_zone import WebcamZoneProcessor


def _frame(center_x: float) -> SimpleNamespace:
    width = 320
    center = round(center_x * width)

    class Boxes(SimpleNamespace):
        def __len__(self):
            return len(self.cls)

    return SimpleNamespace(
        names={0: "person"},
        boxes=Boxes(
            cls=np.array([0]),
            xyxy=np.array([[center - 24, 30, center + 24, 160]], dtype=float),
            id=np.array([7.0]),
        ),
        plot=lambda: np.zeros((180, width, 3), dtype=np.uint8),
    )


def test_webcam_synthetic_marker_uses_camera_frame_and_emits_session_transitions():
    processor = WebcamZoneProcessor(boundary=0.5)
    camera_frame = np.zeros((180, 320, 3), dtype=np.uint8)

    rendered = processor.process_frame(camera_frame, synthetic_center_x=0.2)
    processor.process_frame(camera_frame, synthetic_center_x=0.8)
    processor.process_frame(camera_frame, synthetic_center_x=0.8)
    processor.process_frame(camera_frame, synthetic_center_x=0.2)
    processor.process_frame(camera_frame, synthetic_center_x=0.2)

    assert rendered.shape == camera_frame.shape
    assert [event.event_type for event in processor.events] == [
        "DANGER_ZONE_ENTRY",
        "LEFT_ZONE_ENTRY",
    ]
    assert processor.events[0].track_id == "SIM-01"
    assert processor.events[1].danger_duration_seconds is not None
    assert processor.events[1].danger_duration_seconds >= 0


def test_webcam_real_model_tracks_person_through_same_zone_processor():
    positions = iter((0.2, 0.8, 0.8, 0.2, 0.2))

    class Model:
        def track(self, *, source, persist, conf, verbose):
            assert source.shape == (180, 320, 3)
            assert persist is True and conf == 0.25 and verbose is False
            return [_frame(next(positions))]

    processor = WebcamZoneProcessor(boundary=0.5)
    camera_frame = np.zeros((180, 320, 3), dtype=np.uint8)
    for _ in range(5):
        rendered = processor.process_frame(camera_frame, model=Model())

    assert rendered.shape == camera_frame.shape
    assert [event.event_type for event in processor.events] == [
        "DANGER_ZONE_ENTRY",
        "LEFT_ZONE_ENTRY",
    ]
    assert [event.track_id for event in processor.events] == ["7", "7"]


def test_webcam_processor_rejects_ambiguous_or_invalid_input():
    processor = WebcamZoneProcessor()
    camera_frame = np.zeros((20, 20, 3), dtype=np.uint8)

    try:
        processor.process_frame(camera_frame)
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("Expected missing frame source to be rejected")

    try:
        processor.process_frame(camera_frame, synthetic_center_x=1.1)
    except ValueError as exc:
        assert "between 0 and 1" in str(exc)
    else:
        raise AssertionError("Expected an out-of-range marker position to be rejected")
