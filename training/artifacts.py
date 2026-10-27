"""Artifact discovery for the Model Training page.

This module never invents metrics: absent files are returned as unavailable.
"""

from __future__ import annotations

from pathlib import Path


EXPECTED_ARTIFACTS = ("best.pt", "last.pt", "results.csv", "results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png", "args.yaml")


def scan_run(run_dir: Path) -> dict[str, object]:
    run_dir = Path(run_dir)
    found = {name: str(run_dir / name) for name in EXPECTED_ARTIFACTS if (run_dir / name).exists()}
    return {"run_dir": str(run_dir), "exists": run_dir.exists(), "artifacts": found, "status": "artifacts found" if found else "Experiment not run yet"}
