"""Validated training configuration; no long-running training is auto-started."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    mode: str
    base_model: Path
    dataset_config: Path
    epochs: int
    batch: int
    image_size: int
    device: str
    run_name: str

    def validate(self) -> None:
        if self.mode not in {"full", "demo"}:
            raise ValueError("mode must be full or demo")
        if self.epochs < 1 or self.batch < 1 or self.image_size < 1:
            raise ValueError("training sizes must be positive")
        if not self.base_model.exists():
            raise FileNotFoundError(self.base_model)
        if not self.dataset_config.exists():
            raise FileNotFoundError(self.dataset_config)

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["base_model"] = str(self.base_model)
        values["dataset_config"] = str(self.dataset_config)
        return values
