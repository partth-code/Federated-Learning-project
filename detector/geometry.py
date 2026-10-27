"""Restricted-zone geometry independent of a particular vision model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


Point = tuple[float, float]


def parse_normalized_polygon(value: str) -> list[Point]:
    """Parse ``x,y;x,y;...`` polygon coordinates in the normalized [0, 1] range."""
    try:
        points = [tuple(float(part.strip()) for part in pair.split(",")) for pair in value.split(";") if pair.strip()]
    except (TypeError, ValueError) as exc:
        raise ValueError("Use x,y coordinate pairs separated by semicolons") from exc
    if len(points) < 3 or any(len(point) != 2 for point in points):
        raise ValueError("A restricted zone needs at least three x,y points")
    if any(not 0.0 <= coordinate <= 1.0 for point in points for coordinate in point):
        raise ValueError("Normalized zone coordinates must be between 0 and 1")
    if len(set(points)) < 3:
        raise ValueError("A restricted zone needs at least three distinct points")
    return [(point[0], point[1]) for point in points]


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    if len(polygon) < 3:
        raise ValueError("A restricted zone needs at least three points")
    x, y = point
    inside = False
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        intersects = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1)
        if intersects:
            inside = not inside
    return inside


@dataclass
class ZoneState:
    entered_at: datetime | None = None
    inside: bool = False


class ZoneTracker:
    def __init__(self, zone_id: str, polygon: list[Point], loitering_seconds: float = 0.0) -> None:
        self.zone_id = zone_id
        self.polygon = polygon
        self.loitering_seconds = loitering_seconds
        self._states: dict[str, ZoneState] = {}

    def update(self, person_id: str, footpoint: Point, now: datetime | None = None) -> dict[str, object]:
        now = now or datetime.now(timezone.utc)
        state = self._states.setdefault(person_id, ZoneState())
        inside = point_in_polygon(footpoint, self.polygon)
        entered = inside and not state.inside
        exited = not inside and state.inside
        if entered:
            state.entered_at = now
        if exited:
            state.entered_at = None
        state.inside = inside
        duration = (now - state.entered_at).total_seconds() if inside and state.entered_at else 0.0
        return {
            "zone_id": self.zone_id,
            "person_id": person_id,
            "inside": inside,
            "entered": entered,
            "exited": exited,
            "loitering": inside and duration >= self.loitering_seconds if self.loitering_seconds else False,
            "entered_at": state.entered_at.isoformat() if state.entered_at else None,
            "duration_seconds": duration,
        }
