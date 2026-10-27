"""Technically valid Opacus DP-SGD demonstration on the toy client workload.

This is intentionally separate from the PPE detector and from the basic Flower
run. Its stored scope states exactly what was protected and what was not.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import torch
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

from federated.lightweight_fl import _data_for_client, _model


def run_opacus_demo(db_path: Path, artifact_dir: Path, *, client_index: int = 0, epochs: int = 2, noise_multiplier: float = 1.0, max_grad_norm: float = 1.0, delta: float = 1e-5) -> dict[str, object]:
    if epochs < 1 or noise_multiplier <= 0 or max_grad_norm <= 0 or not 0 < delta < 1:
        raise ValueError("Invalid DP-SGD configuration")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    x, y = _data_for_client(client_index)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=16, shuffle=True)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15)
    privacy_engine = PrivacyEngine()
    model, optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    criterion = torch.nn.CrossEntropyLoss()
    last_loss = None
    for _ in range(epochs):
        for features, labels in private_loader:
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().item())
    epsilon = float(privacy_engine.get_epsilon(delta))
    run_id = f"dp-{uuid.uuid4().hex[:10]}"
    artifact_path = artifact_dir / f"{run_id}.json"
    model_path = artifact_dir / f"{run_id}.pt"
    torch.save(model._module.state_dict(), model_path)
    output = {
        "run_id": run_id,
        "mechanism": "Opacus DP-SGD",
        "scope": "toy local client workload only; not PPE detector and not a SecAgg run",
        "client_id": f"factory-{chr(97 + client_index)}",
        "epochs": epochs,
        "noise_multiplier": noise_multiplier,
        "clipping_norm": max_grad_norm,
        "epsilon": epsilon,
        "delta": delta,
        "last_loss": last_loss,
        "model_artifact": model_path.name,
    }
    artifact_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO privacy_metrics(metric_id, mechanism, epsilon, delta, clipping_norm, noise_multiplier, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, output["mechanism"], epsilon, delta, max_grad_norm, noise_multiplier, output["scope"]),
        )
    return output


if __name__ == "__main__":
    from config import load_settings
    from database import init_db

    settings = load_settings()
    init_db(settings.db_path)
    print(json.dumps(run_opacus_demo(settings.db_path, settings.project_root / "artifacts" / "privacy"), indent=2))
