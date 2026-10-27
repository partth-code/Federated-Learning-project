from pathlib import Path

import pytest

from training.artifacts import scan_run
from training.config import TrainingConfig


def test_training_artifact_scan_is_truthful(tmp_path: Path) -> None:
    assert scan_run(tmp_path)["status"] == "Experiment not run yet"
    (tmp_path / "results.csv").write_text("epoch,metric\n0,0.1\n", encoding="utf-8")
    assert "results.csv" in scan_run(tmp_path)["artifacts"]


def test_training_config_requires_real_inputs(tmp_path: Path) -> None:
    config = TrainingConfig("demo", tmp_path / "missing.pt", tmp_path / "missing.yaml", 1, 1, 32, "cpu", "demo")
    with pytest.raises(FileNotFoundError):
        config.validate()
