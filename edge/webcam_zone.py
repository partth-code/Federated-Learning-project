"""Live OpenCV webcam preview for exercising vertical zone transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import platform
import time
from typing import Any

import numpy as np

from detector.geometry import VerticalZoneTransitionTracker


@dataclass(frozen=True)
class WebcamZoneEvent:
    event_type: str
    track_id: str
    previous_zone: str
    current_zone: str
    elapsed_seconds: float
    danger_duration_seconds: float | None
    timestamp: str


def open_webcam(camera_index: int = 0):
    """Open one local camera, preferring DirectShow on Windows."""
    import cv2

    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_index, backend)
    if not capture.isOpened() and backend != cv2.CAP_ANY:
        capture.release()
        capture = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open webcam {camera_index}. Check that it is connected, not in use by another app, "
            "and permitted by the operating system."
        )
    return capture


class WebcamZoneProcessor:
    """Apply the shared zone tracker to either a synthetic marker or model tracks."""

    def __init__(
        self,
        boundary: float = 0.5,
        hysteresis: float = 0.02,
        confirmations: int = 2,
    ) -> None:
        self.tracker = VerticalZoneTransitionTracker(
            boundary=boundary,
            hysteresis=hysteresis,
            confirmations=confirmations,
            missing_frame_tolerance=2,
        )
        self.started_at = time.monotonic()
        self.danger_started_at: dict[str, float] = {}
        self.events: list[WebcamZoneEvent] = []

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        synthetic_center_x: float | None = None,
        model: Any | None = None,
        confidence: float = 0.25,
    ) -> np.ndarray:
        """Return a zone-annotated BGR frame and retain transitions in memory only."""
        if (synthetic_center_x is None) == (model is None):
            raise ValueError("Provide exactly one of a synthetic position or a PPE model")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Webcam frame must be a three-channel BGR image")
        if synthetic_center_x is not None and not 0.0 <= synthetic_center_x <= 1.0:
            raise ValueError("Synthetic person position must be between 0 and 1")
        if not 0.0 < confidence <= 1.0:
            raise ValueError("Confidence must be greater than 0 and at most 1")

        import cv2

        height, width = frame_bgr.shape[:2]
        if max(height, width) > 1280:
            scale = 1280 / max(height, width)
            frame_bgr = cv2.resize(
                frame_bgr,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            height, width = frame_bgr.shape[:2]

        preview = frame_bgr.copy()
        observations: list[tuple[str, str, tuple[int, int, int, int]]] = []
        transitions: list[tuple[str, str, str]] = []
        self.tracker.begin_frame()
        elapsed = time.monotonic() - self.started_at

        if synthetic_center_x is not None:
            box_width = max(24, round(width * 0.10))
            box_height = max(48, round(height * 0.45))
            center_x = round(synthetic_center_x * width)
            x1 = max(0, center_x - box_width // 2)
            x2 = min(width - 1, center_x + box_width // 2)
            y1 = max(0, round(height * 0.25))
            y2 = min(height - 1, y1 + box_height)
            observation = self.tracker.update("SIM-01", synthetic_center_x)
            observations.append(("SIM-01", str(observation["zone"]), (x1, y1, x2, y2)))
            if observation["transition"] is not None:
                transitions.append((
                    "SIM-01",
                    str(observation["transition"]["previous_zone"]),
                    str(observation["transition"]["current_zone"]),
                ))
            source_label = "SIMULATED PERSON — NO DETECTOR"
        else:
            results = model.track(
                source=frame_bgr,
                persist=True,
                conf=confidence,
                verbose=False,
            )
            source_label = "REAL PPE MODEL"
            if results:
                result = results[0]
                preview = result.plot()
                names = result.names
                boxes = result.boxes
                person_ids = getattr(boxes, "id", None)
                if person_ids is not None:
                    for index in range(len(boxes)):
                        class_id = int(boxes.cls[index].item())
                        label = names[class_id] if isinstance(names, dict) else names[class_id]
                        if str(label).strip().lower() != "person":
                            continue
                        raw_track_id = person_ids[index].item()
                        try:
                            numeric_track_id = float(raw_track_id)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(numeric_track_id):
                            continue
                        track_id = str(int(numeric_track_id))
                        x1, y1, x2, y2 = (float(value) for value in boxes.xyxy[index].tolist())
                        center_x = ((x1 + x2) / 2) / width
                        observation = self.tracker.update(track_id, center_x)
                        observations.append((
                            track_id,
                            str(observation["zone"]),
                            (round(x1), round(y1), round(x2), round(y2)),
                        ))
                        if observation["transition"] is not None:
                            transitions.append((
                                track_id,
                                str(observation["transition"]["previous_zone"]),
                                str(observation["transition"]["current_zone"]),
                            ))

        self.tracker.end_frame()
        overlay = preview.copy()
        boundary_x = round(self.tracker.boundary * width)
        cv2.rectangle(overlay, (0, 0), (boundary_x, height), (40, 150, 40), -1)
        cv2.rectangle(overlay, (boundary_x, 0), (width, height), (40, 40, 190), -1)
        preview = cv2.addWeighted(overlay, 0.14, preview, 0.86, 0)
        cv2.line(preview, (boundary_x, 0), (boundary_x, height), (255, 255, 255), 2)
        cv2.putText(preview, "LEFT ZONE", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 0), 2)
        cv2.putText(
            preview, "DANGER ZONE", (min(width - 180, boundary_x + 12), 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2,
        )
        cv2.putText(preview, source_label, (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        for track_id, zone, (x1, y1, x2, y2) in observations:
            color = (0, 160, 0) if zone == "LEFT" else (0, 0, 220)
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            label = f"{'SIMULATED ' if track_id == 'SIM-01' else ''}ID {track_id} / {zone}"
            cv2.putText(
                preview, label, (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        for track_id, previous_zone, current_zone in transitions:
            if current_zone == "RIGHT":
                self.danger_started_at[track_id] = elapsed
                duration = None
                event_type = "DANGER_ZONE_ENTRY"
            else:
                started = self.danger_started_at.pop(track_id, None)
                duration = max(0.0, elapsed - started) if started is not None else None
                event_type = "LEFT_ZONE_ENTRY"
            self.events.append(WebcamZoneEvent(
                event_type=event_type,
                track_id=track_id,
                previous_zone=previous_zone,
                current_zone=current_zone,
                elapsed_seconds=elapsed,
                danger_duration_seconds=duration,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))

        return preview
