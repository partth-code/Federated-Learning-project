"""Bounded local frame sampling for user-provided video clips."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

import numpy as np

from detector.contracts import Detection, normalize_ppe_label
from detector.geometry import Point, ZoneTracker


@dataclass(frozen=True)
class VideoFrameResult:
    frame_number: int
    elapsed_seconds: float
    detections: tuple[Detection, ...]
    preview_jpeg: bytes
    evidence_jpeg: bytes
    zone_entries: tuple["VideoZoneEntry", ...] = ()


@dataclass(frozen=True)
class VideoZoneEntry:
    detection: Detection
    zone_id: str
    entered_at: str
    duration_seconds: float


@dataclass(frozen=True)
class VideoAnalysis:
    frames: tuple[VideoFrameResult, ...]
    fps: float
    duration_seconds: float
    truncated: bool


def record_selected_video_frame(
    *,
    db_path: Path,
    frame: VideoFrameResult,
    site_id: str,
    camera_id: str,
    evidence_dir: Path,
    federation_online: bool,
):
    """Persist only model-returned PPE violations from a reviewer-selected frame."""
    from edge.incident_pipeline import PPE_VIOLATIONS, record_detection_events

    violations = [item for item in frame.detections if item.label.strip().lower() in PPE_VIOLATIONS]
    if not violations:
        return []
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(frame.evidence_jpeg)
            temporary_path = Path(handle.name)
        return record_detection_events(
            db_path=db_path,
            detections=violations,
            site_id=site_id,
            camera_id=camera_id,
            evidence_source=temporary_path,
            evidence_dir=evidence_dir,
            federation_online=federation_online,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def record_selected_video_zone_entry(
    *,
    db_path: Path,
    frame: VideoFrameResult,
    entry: VideoZoneEntry,
    site_id: str,
    camera_id: str,
    evidence_dir: Path,
    federation_online: bool,
):
    """Persist one tracked-person zone entry selected from a sampled frame."""
    from edge.incident_pipeline import record_zone_entry

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(frame.evidence_jpeg)
            temporary_path = Path(handle.name)
        incident = record_zone_entry(
            db_path=db_path,
            detection=entry.detection,
            site_id=site_id,
            camera_id=camera_id,
            zone_id=entry.zone_id,
            entered_at=entry.entered_at,
            duration_seconds=entry.duration_seconds,
            evidence_source=temporary_path,
            evidence_dir=evidence_dir,
            federation_online=federation_online,
        )
        return [incident] if incident is not None else []
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def analyze_local_video(
    video_path: Path,
    model,
    *,
    interval_seconds: float = 1.0,
    max_frames: int = 30,
    max_dimension: int = 1280,
    confidence: float = 0.25,
    zone_polygon: list[Point] | None = None,
    zone_id: str = "restricted-demo-zone",
) -> VideoAnalysis:
    """Analyze at most ``max_frames`` sampled frames, entirely on this machine.

    Only JPEG previews and still evidence bytes are returned; the uploaded source
    video is never copied into the database or evidence directory by this helper.
    The caller owns and should remove the temporary source file.
    """
    if interval_seconds <= 0 or max_frames <= 0 or max_dimension <= 0:
        raise ValueError("Sampling interval, frame limit, and max dimension must be positive")

    import cv2

    zone_tracker = ZoneTracker(zone_id, zone_polygon) if zone_polygon is not None else None
    seen_track_ids: set[str] = set()
    clip_started_at = datetime.now(timezone.utc)
    person_class_ids: set[int] = set()
    if zone_tracker is not None:
        names = model.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        person_class_ids = {
            int(class_id)
            for class_id, label in items
            if str(label).strip().lower() == "person"
        }
        if not person_class_ids:
            raise ValueError("The configured PPE model has no person class for zone tracking")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Video could not be opened by the local decoder")

    frames: list[VideoFrameResult] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no usable frame-rate metadata")
        reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = reported_frame_count / fps if reported_frame_count > 0 else 0.0
        step = max(1, round(fps * interval_seconds))
        frame_number = 0

        while len(frames) < max_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_number % step == 0:
                height, width = frame_bgr.shape[:2]
                scale = min(1.0, max_dimension / max(height, width))
                if scale < 1.0:
                    frame_bgr = cv2.resize(
                        frame_bgr,
                        (max(1, round(width * scale)), max(1, round(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                if zone_tracker is None:
                    prediction_results = model.predict(source=frame_bgr, conf=confidence, verbose=False)
                else:
                    prediction_results = model.track(
                        source=frame_bgr,
                        persist=True,
                        conf=confidence,
                        verbose=False,
                    )
                zone_entries: list[VideoZoneEntry] = []
                if not prediction_results:
                    detections: tuple[Detection, ...] = ()
                    preview_bgr = frame_bgr
                else:
                    result = prediction_results[0]
                    timestamp = datetime.now(timezone.utc).isoformat()
                    values: list[Detection] = []
                    person_ids = getattr(result.boxes, "id", None)
                    for index in range(len(result.boxes)):
                        class_id = int(result.boxes.cls[index].item())
                        coords = tuple(float(value) for value in result.boxes.xyxy[index].tolist())
                        track_id = None
                        if zone_tracker is not None and person_ids is not None:
                            candidate_id = person_ids[index].item()
                            if int(class_id) in person_class_ids and candidate_id is not None:
                                track_id = str(int(candidate_id))
                        detection = Detection(
                            normalize_ppe_label(result.names[class_id]),
                            float(result.boxes.conf[index].item()),
                            coords,
                            timestamp,
                            track_id=track_id,
                        )
                        values.append(detection)
                        if zone_tracker is not None and track_id is not None:
                            height, width = frame_bgr.shape[:2]
                            x1, _, x2, y2 = detection.bbox_xyxy
                            elapsed = frame_number / fps
                            observation = zone_tracker.update(
                                track_id,
                                ((x1 + x2) / (2 * width), y2 / height),
                                now=clip_started_at + timedelta(seconds=elapsed),
                            )
                            first_observation = track_id not in seen_track_ids
                            seen_track_ids.add(track_id)
                            if observation["entered"] and not first_observation:
                                zone_entries.append(VideoZoneEntry(
                                    detection=detection,
                                    zone_id=str(observation["zone_id"]),
                                    entered_at=str(observation["entered_at"]),
                                    duration_seconds=float(observation["duration_seconds"]),
                                ))
                    detections = tuple(values)
                    preview_bgr = result.plot()

                if zone_tracker is not None:
                    height, width = frame_bgr.shape[:2]
                    polygon_pixels = [
                        (round(x * width), round(y * height))
                        for x, y in zone_tracker.polygon
                    ]
                    cv2.polylines(preview_bgr, [np.asarray(polygon_pixels, dtype=np.int32)], True, (0, 165, 255), 2)

                evidence_ok, evidence = cv2.imencode(".jpg", frame_bgr)
                preview_ok, preview = cv2.imencode(".jpg", preview_bgr)
                if not evidence_ok or not preview_ok:
                    raise ValueError("Local decoder could not encode a sampled frame")
                frames.append(VideoFrameResult(
                    frame_number=frame_number,
                    elapsed_seconds=frame_number / fps,
                    detections=detections,
                    preview_jpeg=preview.tobytes() if isinstance(preview, np.ndarray) else bytes(preview),
                    evidence_jpeg=evidence.tobytes() if isinstance(evidence, np.ndarray) else bytes(evidence),
                    zone_entries=tuple(zone_entries),
                ))
            frame_number += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError("Video contained no decodable frames")
    truncated = len(frames) >= max_frames and (
        duration <= 0 or duration > (frames[-1].elapsed_seconds + interval_seconds)
    )
    return VideoAnalysis(tuple(frames), fps, duration, truncated)
