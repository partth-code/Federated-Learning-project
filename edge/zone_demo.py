"""Deterministic, in-memory zone flow for UI testing without a PPE model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from PIL import Image, ImageDraw

from detector.geometry import VerticalZoneTransitionTracker


@dataclass(frozen=True)
class SimulatedZoneFrame:
    sample: int
    elapsed_seconds: float
    zone: str
    annotated_image: Image.Image


@dataclass(frozen=True)
class SimulatedZoneEvent:
    event_type: str
    track_id: str
    previous_zone: str
    current_zone: str
    elapsed_seconds: float
    danger_duration_seconds: float | None
    timestamp: str


@dataclass(frozen=True)
class SimulatedZoneRun:
    frames: tuple[SimulatedZoneFrame, ...]
    events: tuple[SimulatedZoneEvent, ...]
    boundary: float


def run_simulated_zone_demo(boundary: float = 0.5) -> SimulatedZoneRun:
    """Render a fixed left→right→left sequence; never writes incidents or evidence."""
    tracker = VerticalZoneTransitionTracker(
        boundary=boundary,
        hysteresis=0.02,
        confirmations=2,
        missing_frame_tolerance=0,
    )
    sample_positions = (0.2, 0.2, 0.8, 0.8, 0.8, 0.2, 0.2)
    sample_interval = 1.0
    danger_started_at: float | None = None
    frames: list[SimulatedZoneFrame] = []
    events: list[SimulatedZoneEvent] = []
    started_at = datetime.now(timezone.utc)

    for sample, position in enumerate(sample_positions):
        elapsed = sample * sample_interval
        tracker.begin_frame()
        observation = tracker.update("SIM-01", position)
        tracker.end_frame()
        zone = str(observation["zone"])
        image = Image.new("RGB", (800, 450), "#f8fafc")
        draw = ImageDraw.Draw(image, "RGBA")
        boundary_x = round(boundary * image.width)
        draw.rectangle((0, 0, boundary_x, image.height), fill=(42, 170, 80, 38))
        draw.rectangle((boundary_x, 0, image.width, image.height), fill=(220, 48, 48, 38))
        draw.line((boundary_x, 0, boundary_x, image.height), fill="#26364a", width=4)
        draw.text((20, 18), "SIMULATED LEFT ZONE", fill="#176b35")
        draw.text((boundary_x + 16, 18), "SIMULATED DANGER ZONE", fill="#ad2424")
        person_x = round(position * image.width)
        box = (person_x - 42, 150, person_x + 42, 350)
        color = "#19803c" if zone == "LEFT" else "#c62828"
        draw.rectangle(box, outline=color, width=5)
        draw.text((box[0], box[1] - 24), f"SIMULATED ID SIM-01 · {zone}", fill=color)
        draw.text((20, 420), f"Synthetic sample {sample + 1} · clip offset {elapsed:.1f}s", fill="#334155")
        frames.append(SimulatedZoneFrame(sample, elapsed, zone, image))

        transition = observation["transition"]
        if transition is None:
            continue
        previous_zone = str(transition["previous_zone"])
        current_zone = str(transition["current_zone"])
        if current_zone == "RIGHT":
            danger_started_at = elapsed
            event_type = "DANGER_ZONE_ENTRY"
            duration = None
        else:
            event_type = "LEFT_ZONE_ENTRY"
            duration = max(0.0, elapsed - danger_started_at) if danger_started_at is not None else None
            danger_started_at = None
        events.append(SimulatedZoneEvent(
            event_type=event_type,
            track_id="SIM-01",
            previous_zone=previous_zone,
            current_zone=current_zone,
            elapsed_seconds=elapsed,
            danger_duration_seconds=duration,
            timestamp=(started_at + timedelta(seconds=elapsed)).isoformat(),
        ))

    return SimulatedZoneRun(tuple(frames), tuple(events), boundary)
