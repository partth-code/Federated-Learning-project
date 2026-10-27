"""Bounded local frame sampling for user-provided video clips."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

from detector.contracts import Detection, normalize_ppe_label
from audit.hash_chain import append_audit
from database import insert_incident
from detector.geometry import Point, VerticalZoneTransitionTracker, ZoneTracker
from edge.evidence import store_local_evidence
from edge.offline_queue import enqueue
from incidents.schema import Incident


@dataclass(frozen=True)
class VideoFrameResult:
    frame_number: int
    elapsed_seconds: float
    detections: tuple[Detection, ...]
    preview_jpeg: bytes
    evidence_jpeg: bytes
    zone_entries: tuple["VideoZoneEntry", ...] = ()
    zone_transitions: tuple["VideoZoneTransition", ...] = ()
    tracked_zones: tuple[tuple[str, str], ...] = ()
    left_count: int = 0
    danger_count: int = 0


@dataclass(frozen=True)
class VideoZoneEntry:
    detection: Detection
    zone_id: str
    entered_at: str
    duration_seconds: float


@dataclass(frozen=True)
class VideoZoneTransition:
    detection: Detection
    previous_zone: str
    current_zone: str
    elapsed_seconds: float
    danger_duration_seconds: float | None = None


@dataclass(frozen=True)
class VideoAnalysis:
    frames: tuple[VideoFrameResult, ...]
    fps: float
    duration_seconds: float
    truncated: bool
    zone_boundary: float | None = None
    sample_interval_seconds: float = 1.0


def record_video_zone_transitions(
    *,
    db_path: Path,
    frame: VideoFrameResult,
    site_id: str,
    camera_id: str,
    evidence_dir: Path,
    federation_online: bool,
    source_key_prefix: str,
) -> list[Incident]:
    """Persist confirmed left/danger transitions and resolve the matching entry."""
    created: list[Incident] = []
    for transition in frame.zone_transitions:
        entering_danger = transition.current_zone == "RIGHT"
        event_type = "DANGER_ZONE_ENTRY" if entering_danger else "LEFT_ZONE_ENTRY"
        source_key = hashlib.sha256(
            f"{source_key_prefix}:{camera_id}:{transition.detection.track_id}:"
            f"{event_type}:{transition.elapsed_seconds:.3f}".encode("utf-8")
        ).hexdigest()
        with sqlite3.connect(db_path) as conn:
            if conn.execute(
                "SELECT 1 FROM incidents WHERE source_key = ?", (source_key,)
            ).fetchone():
                continue

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
                handle.write(frame.preview_jpeg)
                temporary_path = Path(handle.name)
            evidence_path = str(store_local_evidence(temporary_path, evidence_dir, source_key))
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        incident = Incident(
            event_id=source_key[:8] + "-" + source_key[8:12] + "-" + source_key[12:16]
            + "-" + source_key[16:20] + "-" + source_key[20:32],
            site_id=site_id,
            camera_id=camera_id,
            timestamp=transition.detection.timestamp,
            event_type=event_type,
            person_id=transition.detection.track_id,
            confidence=transition.detection.confidence,
            zone_id="vertical-danger-zone",
            zone_entered_at=transition.detection.timestamp if entering_danger else None,
            zone_duration_seconds=transition.danger_duration_seconds,
            evidence_path=evidence_path,
            previous_zone=transition.previous_zone,
            current_zone=transition.current_zone,
            video_timestamp_seconds=transition.elapsed_seconds,
            zone_exited_at=None if entering_danger else transition.detection.timestamp,
            zone_status="active" if entering_danger else "resolved",
            severity="medium",
            source_key=source_key,
        )
        if not insert_incident(db_path, incident):
            continue
        with sqlite3.connect(db_path) as conn:
            if not entering_danger:
                active_entry = conn.execute(
                    """SELECT event_id FROM incidents
                    WHERE site_id = ? AND camera_id = ? AND person_id = ?
                      AND event_type = 'DANGER_ZONE_ENTRY' AND zone_status = 'active'
                    ORDER BY timestamp DESC, created_at DESC LIMIT 1""",
                    (site_id, camera_id, transition.detection.track_id),
                ).fetchone()
                if active_entry:
                    conn.execute(
                        """UPDATE incidents
                        SET zone_status = 'resolved', zone_exited_at = ?, zone_duration_seconds = ?
                        WHERE event_id = ?""",
                        (incident.zone_exited_at, transition.danger_duration_seconds, active_entry[0]),
                    )
        if not entering_danger and active_entry:
            append_audit(db_path, "incident.zone_resolved", {
                "event_id": active_entry[0],
                "exit_event_id": incident.event_id,
                "duration_seconds": transition.danger_duration_seconds,
            })
        if not federation_online:
            enqueue(db_path, "incident_metadata", incident.event_id, {
                "event_id": incident.event_id,
                "site_id": site_id,
                "camera_id": camera_id,
                "event_type": incident.event_type,
            })
        created.append(incident)
    return created


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
    zone_boundary: float | None = None,
    zone_hysteresis: float = 0.02,
    transition_confirmations: int = 2,
    missing_frame_tolerance: int = 2,
) -> VideoAnalysis:
    """Analyze at most ``max_frames`` sampled frames, entirely on this machine.

    Only JPEG previews and still evidence bytes are returned; the uploaded source
    video is never copied into the database or evidence directory by this helper.
    The caller owns and should remove the temporary source file.
    """
    if interval_seconds <= 0 or max_frames <= 0 or max_dimension <= 0:
        raise ValueError("Sampling interval, frame limit, and max dimension must be positive")

    import cv2

    if zone_polygon is not None and zone_boundary is not None:
        raise ValueError("Choose either a polygon zone or the vertical left/danger zones")
    zone_tracker = ZoneTracker(zone_id, zone_polygon) if zone_polygon is not None else None
    transition_tracker = (
        VerticalZoneTransitionTracker(
            zone_boundary,
            hysteresis=zone_hysteresis,
            confirmations=transition_confirmations,
            missing_frame_tolerance=missing_frame_tolerance,
        )
        if zone_boundary is not None else None
    )
    seen_track_ids: set[str] = set()
    danger_started_at: dict[str, float] = {}
    clip_started_at = datetime.now(timezone.utc)
    person_class_ids: set[int] = set()
    if zone_tracker is not None or transition_tracker is not None:
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
                if zone_tracker is None and transition_tracker is None:
                    prediction_results = model.predict(source=frame_bgr, conf=confidence, verbose=False)
                else:
                    prediction_results = model.track(
                        source=frame_bgr,
                        persist=True,
                        conf=confidence,
                        verbose=False,
                    )
                zone_entries: list[VideoZoneEntry] = []
                zone_transitions: list[VideoZoneTransition] = []
                tracked_zones: list[tuple[str, str]] = []
                if transition_tracker is not None:
                    transition_tracker.begin_frame()
                if not prediction_results:
                    detections: tuple[Detection, ...] = ()
                    preview_bgr = frame_bgr
                else:
                    result = prediction_results[0]
                    elapsed = frame_number / fps
                    timestamp = datetime.now(timezone.utc).isoformat()
                    values: list[Detection] = []
                    person_ids = getattr(result.boxes, "id", None)
                    for index in range(len(result.boxes)):
                        class_id = int(result.boxes.cls[index].item())
                        coords = tuple(float(value) for value in result.boxes.xyxy[index].tolist())
                        track_id = None
                        if (zone_tracker is not None or transition_tracker is not None) and person_ids is not None:
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
                        if transition_tracker is not None and track_id is not None:
                            _, _, x2, _ = detection.bbox_xyxy
                            x1 = detection.bbox_xyxy[0]
                            observation = transition_tracker.update(
                                track_id,
                                ((x1 + x2) / 2) / width,
                            )
                            zone = str(observation["zone"])
                            tracked_zones.append((track_id, zone))
                            transition = observation["transition"]
                            if transition is not None:
                                previous_zone = str(transition["previous_zone"])
                                current_zone = str(transition["current_zone"])
                                danger_duration = None
                                if current_zone == "RIGHT":
                                    danger_started_at[track_id] = elapsed
                                elif track_id in danger_started_at:
                                    danger_duration = max(0.0, elapsed - danger_started_at.pop(track_id))
                                    zone_transitions.append(VideoZoneTransition(
                                        detection=detection,
                                        previous_zone=previous_zone,
                                        current_zone=current_zone,
                                        elapsed_seconds=elapsed,
                                        danger_duration_seconds=danger_duration,
                                    ))
                                if current_zone == "RIGHT":
                                    zone_transitions.append(VideoZoneTransition(
                                        detection=detection,
                                        previous_zone=previous_zone,
                                        current_zone=current_zone,
                                        elapsed_seconds=elapsed,
                                    ))
                    detections = tuple(values)
                    preview_bgr = result.plot()

                if transition_tracker is not None:
                    transition_tracker.end_frame()
                    overlay = preview_bgr.copy()
                    boundary_x = round(transition_tracker.boundary * width)
                    cv2.rectangle(overlay, (0, 0), (boundary_x, height), (40, 150, 40), -1)
                    cv2.rectangle(overlay, (boundary_x, 0), (width, height), (40, 40, 190), -1)
                    preview_bgr = cv2.addWeighted(overlay, 0.14, preview_bgr, 0.86, 0)
                    cv2.line(preview_bgr, (boundary_x, 0), (boundary_x, height), (255, 255, 255), 2)
                    cv2.putText(preview_bgr, "LEFT ZONE", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 0), 2)
                    cv2.putText(preview_bgr, "DANGER ZONE", (boundary_x + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2)
                    tracked_by_id = dict(tracked_zones)
                    for detection in detections:
                        if detection.label != "person" or detection.track_id is None:
                            continue
                        x1, y1, x2, y2 = (round(value) for value in detection.bbox_xyxy)
                        zone = tracked_by_id[detection.track_id]
                        color = (0, 160, 0) if zone == "LEFT" else (0, 0, 220)
                        cv2.rectangle(preview_bgr, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            preview_bgr, f"ID {detection.track_id} / {zone}",
                            (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                        )
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
                    zone_transitions=tuple(zone_transitions),
                    tracked_zones=tuple(tracked_zones),
                    left_count=sum(zone == "LEFT" for _, zone in tracked_zones),
                    danger_count=sum(zone == "RIGHT" for _, zone in tracked_zones),
                ))
            frame_number += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError("Video contained no decodable frames")
    truncated = len(frames) >= max_frames and (
        duration <= 0 or duration > (frames[-1].elapsed_seconds + interval_seconds)
    )
    return VideoAnalysis(tuple(frames), fps, duration, truncated, zone_boundary, interval_seconds)
