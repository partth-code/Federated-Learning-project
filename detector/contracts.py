"""Model-independent detection contracts.

The real PPE adapter is intentionally gated on a verified PPE model. A generic
COCO detector must never be mislabeled as helmet/vest detection.
"""

from __future__ import annotations

from dataclasses import dataclass


class DetectorUnavailable(RuntimeError):
    """Raised when a verified PPE model is not configured or available."""


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    timestamp: str
    track_id: str | None = None


SUPPORTED_PPE_LABELS = frozenset({
    "person", "helmet", "no_helmet", "vest", "no_vest", "gloves", "no_gloves",
    "boots", "no_boots", "goggles", "no_goggle", "none",
    "dust_mask", "shield",
})


def normalize_ppe_label(label: str) -> str:
    """Explicit synonyms in the supplied PPE checkpoint; never infer absence."""
    value = str(label).strip().lower()
    return {"dust mask": "dust_mask", "eye wear": "goggles", "glove": "gloves"}.get(value, value)


def validate_ppe_labels(labels: set[str]) -> None:
    labels = {normalize_ppe_label(label) for label in labels}
    unknown = labels - SUPPORTED_PPE_LABELS
    if unknown:
        raise ValueError(f"Unsupported PPE labels: {sorted(unknown)}")
    if not labels.intersection({"helmet", "no_helmet", "vest", "no_vest"}):
        raise DetectorUnavailable("Configured model has no verified PPE classes; COCO person is not PPE detection")
