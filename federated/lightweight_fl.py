"""A real, small Flower federated-learning integration test.

This workload is deliberately not the PPE detector. It uses a two-feature
PyTorch classifier and deterministic non-IID synthetic client data so the
federation plumbing can be tested without inventing PPE metrics.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import socket
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from flwr.client import NumPyClient, start_client
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg

from models.registry import register_version


CLIENTS = ("factory-a", "factory-b", "factory-c")


def _data_for_client(client_index: int, n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1000 + client_index)
    # Each site sees a different, deterministic feature shift: non-IID but reproducible.
    shift = np.array([client_index * 0.8, -client_index * 0.4], dtype=np.float32)
    x = rng.normal(size=(n, 2)).astype(np.float32) + shift
    y = ((x[:, 0] + 0.7 * x[:, 1]) > 0).astype(np.int64)
    return x, y


def _held_out_data(n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9000)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    y = ((x[:, 0] + 0.7 * x[:, 1]) > 0).astype(np.int64)
    return x, y


def _model() -> torch.nn.Module:
    torch.manual_seed(7)
    return torch.nn.Linear(2, 2)


def _get_parameters(model: torch.nn.Module) -> NDArrays:
    return [value.detach().cpu().numpy() for _, value in model.state_dict().items()]


def _set_parameters(model: torch.nn.Module, parameters: NDArrays) -> None:
    state = model.state_dict()
    for (name, _), value in zip(state.items(), parameters):
        state[name] = torch.tensor(value)
    model.load_state_dict(state)


def _train(model: torch.nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15)
    criterion = torch.nn.CrossEntropyLoss()
    features = torch.from_numpy(x)
    labels = torch.from_numpy(y)
    for _ in range(3):
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()
    return float(loss.detach().item())


def _evaluate(model: torch.nn.Module, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y)).item()
        accuracy = (logits.argmax(dim=1).numpy() == y).mean()
    return float(loss), float(accuracy)


class _SafetyToyClient(NumPyClient):
    def __init__(self, client_index: int) -> None:
        self.client_index = client_index
        self.site_id = CLIENTS[client_index]
        self.x, self.y = _data_for_client(client_index)
        self.model = _model()

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return _get_parameters(self.model)

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
        _set_parameters(self.model, parameters)
        loss = _train(self.model, self.x, self.y)
        return _get_parameters(self.model), len(self.y), {"local_loss": loss}

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]:
        _set_parameters(self.model, parameters)
        loss, accuracy = _evaluate(self.model, self.x, self.y)
        return loss, len(self.y), {"accuracy": accuracy}


def _client_process(client_index: int, server_address: str) -> None:
    client = _SafetyToyClient(client_index).to_client()
    start_client(server_address=server_address, client=client)


def _server_process(server_address: str, rounds: int, result_queue: mp.Queue, model_path: str) -> None:
    global_model = _model()
    x_test, y_test = _held_out_data()

    def evaluate_fn(server_round: int, parameters, config: dict[str, Scalar]):
        model = _model()
        _set_parameters(model, [array for array in parameters])
        loss, accuracy = _evaluate(model, x_test, y_test)
        if server_round == rounds:
            torch.save(model.state_dict(), model_path)
        return loss, {"accuracy": accuracy}

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=len(CLIENTS),
        min_evaluate_clients=0,
        min_available_clients=len(CLIENTS),
        initial_parameters=ndarrays_to_parameters(_get_parameters(global_model)),
        evaluate_fn=evaluate_fn,
    )
    history = start_server(
        server_address=server_address,
        config=ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )
    metrics = {str(round_number): values for round_number, values in history.metrics_centralized.get("accuracy", [])}
    result_queue.put({"round_metrics": metrics, "loss_metrics": history.losses_centralized})


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_federated(db_path: Path, artifact_dir: Path, rounds: int = 3) -> dict[str, object]:
    if rounds < 1:
        raise ValueError("rounds must be positive")
    db_path = Path(db_path)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"fl-{uuid.uuid4().hex[:10]}"
    model_path = artifact_dir / f"{run_id}.pt"
    artifact_path = artifact_dir / f"{run_id}.json"
    port = _free_port()
    address = f"127.0.0.1:{port}"
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    server = context.Process(target=_server_process, args=(address, rounds, result_queue, str(model_path)))
    server.start()
    time.sleep(1.0)
    clients = [context.Process(target=_client_process, args=(index, address)) for index in range(len(CLIENTS))]
    for client in clients:
        client.start()
    for client in clients:
        client.join(timeout=120)
    server.join(timeout=120)
    if server.is_alive() or any(client.is_alive() for client in clients):
        for process in [server, *clients]:
            if process.is_alive():
                process.terminate()
        raise RuntimeError("Flower federation timed out")
    if server.exitcode != 0 or any(client.exitcode != 0 for client in clients):
        raise RuntimeError(f"Flower federation failed: server={server.exitcode}, clients={[c.exitcode for c in clients]}")
    result = result_queue.get(timeout=10)
    if not model_path.exists():
        raise RuntimeError("Flower completed but did not produce a final global model artifact")
    output = {
        "run_id": run_id,
        "workload": "lightweight PyTorch classifier; not PPE detector",
        "simulated_sites": list(CLIENTS),
        "rounds": rounds,
        "round_metrics": result["round_metrics"],
        "loss_metrics": result["loss_metrics"],
        "model_artifact": model_path.name,
        "secure_aggregation": False,
        "differential_privacy": False,
    }
    artifact_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    version_id = register_version(
        db_path,
        model_path,
        round_number=rounds,
        validation_metric=float(result["round_metrics"][str(rounds)]),
        notes="Real Flower lightweight integration test; not a PPE detector",
        model_kind="toy_classifier",
        clients=CLIENTS,
        project_root=artifact_dir.parent.parent,
    )
    output["model_version_id"] = version_id
    artifact_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        for round_number, accuracy in result["round_metrics"].items():
            conn.execute(
                "INSERT INTO federation_rounds(round_id, round_number, status, metrics_json) VALUES (?, ?, ?, ?)",
                (f"{run_id}-r{round_number}", int(round_number), "completed", json.dumps({"accuracy": accuracy})),
            )
        conn.commit()
    return output


if __name__ == "__main__":
    from config import load_settings
    from database import init_db

    settings = load_settings()
    init_db(settings.db_path)
    print(json.dumps(run_federated(settings.db_path, settings.project_root / "artifacts" / "federated"), indent=2))
