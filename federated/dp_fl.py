"""Experimental example-level DP-SGD in three real Flower toy clients.

Accountants persist within a run, not across runs. No PPE or SecAgg claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import socket
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from flwr.client import NumPyClient, start_client
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

from federated.lightweight_fl import (
    CLIENTS, _data_for_client, _evaluate, _free_port, _get_parameters,
    _held_out_data, _model,
)
from models.registry import register_version


@dataclass(frozen=True)
class DPConfig:
    epochs: int = 1
    noise_multiplier: float = 1.0
    clipping_norm: float = 1.0
    delta: float = 1e-5

    def validate(self):
        if type(self.epochs) is not int or not 1 <= self.epochs <= 2:
            raise ValueError("Demo epochs must be 1 or 2")
        if any(not math.isfinite(v) or v <= 0 for v in (self.noise_multiplier, self.clipping_norm)):
            raise ValueError("Noise and clipping must be finite and positive")
        if not math.isfinite(self.delta) or not 0 < self.delta < 1 / 80:
            raise ValueError("Delta must be positive and below 1/client sample count")


def set_parameters(model, parameters):
    state = model.state_dict()
    if len(parameters) != len(state):
        raise ValueError("Parameter count mismatch")
    checked = {}
    for (name, tensor), array in zip(state.items(), parameters):
        array = np.asarray(array)
        if array.shape != tuple(tensor.shape) or array.dtype != tensor.numpy().dtype or not np.isfinite(array).all():
            raise ValueError("Invalid parameter shape, dtype or value")
        checked[name] = torch.from_numpy(array.copy())
    model.load_state_dict(checked, strict=True)


class DPClient(NumPyClient):
    def __init__(self, index: int, config: DPConfig):
        config.validate()
        self.config = config
        self.site = CLIENTS[index]
        x, y = _data_for_client(index)
        # Isolate deterministic initialization from all sampling/noise RNG state.
        with torch.random.fork_rng(devices=[]):
            base = _model()
        noise = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), 'big'))
        sampling = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), 'big'))
        loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                            batch_size=16, generator=sampling)
        optimizer = torch.optim.SGD(base.parameters(), lr=0.15)
        self.engine = PrivacyEngine(accountant="rdp", secure_mode=False)
        self.model, self.optimizer, self.loader = self.engine.make_private(
            module=base, optimizer=optimizer, data_loader=loader,
            noise_multiplier=config.noise_multiplier, max_grad_norm=config.clipping_norm,
            noise_generator=noise, poisson_sampling=True,
        )
        self.steps = 0
        self.round = 0

    def get_parameters(self, config):
        return [a.copy() for a in _get_parameters(self.model._module)]

    def fit(self, parameters, config):
        expected = self.round + 1
        if int(config.get("server_round", -1)) != expected:
            raise ValueError("Out-of-order/replayed round; accountant must not reset")
        set_parameters(self.model._module, parameters)
        self.model.train()
        for _ in range(self.config.epochs):
            for x, y in self.loader:
                self.optimizer.zero_grad()
                # Sum / max handles empty Poisson batches without skipping noise.
                logits = self.model(x)
                loss = torch.nn.functional.cross_entropy(logits, y, reduction="sum") / max(1, len(y))
                loss.backward()
                self.optimizer.step()
                self.steps += 1
        self.round = expected
        metrics = {
            "site_id": self.site, "round": self.round, "steps": self.steps,
            "epsilon": float(self.engine.get_epsilon(self.config.delta)),
            "delta": self.config.delta, "sample_rate": self.loader.sample_rate,
        }
        # 80 is the public fixed synthetic dataset size, not a private count.
        return self.get_parameters({}), 80, metrics

    def evaluate(self, parameters, config):
        raise RuntimeError("Local private-data evaluation is disabled")


def _client(index, address, config):
    torch.set_num_threads(1)
    start_client(server_address=address, client=DPClient(index, config).to_client())


def _server(address, rounds, model_path, result_path):
    torch.set_num_threads(1)
    model = _model()
    x, y = _held_out_data()
    records = []

    class StrictFedAvg(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            if failures or len(results) != 3:
                raise RuntimeError("All three clients must complete each DP round")
            metrics = [dict(result.metrics) for _, result in results]
            if {m["site_id"] for m in metrics} != set(CLIENTS):
                raise RuntimeError("Missing/duplicate DP clients")
            if any(m["round"] != server_round or not math.isfinite(m["epsilon"]) for m in metrics):
                raise RuntimeError("Invalid DP accounting evidence")
            aggregate = super().aggregate_fit(server_round, results, failures)
            records.append({"round": server_round, "clients": sorted(metrics, key=lambda m: m["site_id"])})
            return aggregate

    def evaluate(round_number, parameters, config):
        set_parameters(model, parameters)
        loss, accuracy = _evaluate(model, x, y)
        if round_number == rounds:
            torch.save(model.state_dict(), model_path)
        return loss, {"accuracy": accuracy}

    history = start_server(server_address=address, config=ServerConfig(num_rounds=rounds, round_timeout=90),
        strategy=StrictFedAvg(fraction_fit=1, fraction_evaluate=0, min_fit_clients=3,
            min_available_clients=3, min_evaluate_clients=0, accept_failures=False,
            initial_parameters=ndarrays_to_parameters(_get_parameters(model)),
            on_fit_config_fn=lambda r: {"server_round": r}, evaluate_fn=evaluate))
    if len(records) != rounds:
        raise RuntimeError("Incomplete DP federation")
    Path(result_path).write_text(json.dumps({"accounting": records,
        "accuracy": dict(history.metrics_centralized["accuracy"]),
        "loss": dict(history.losses_centralized)}), encoding="utf-8")


def run_dp_federated(db_path: Path, artifact_dir: Path, rounds: int = 2,
                     config: DPConfig = DPConfig()):
    config.validate()
    if type(rounds) is not int or not 1 <= rounds <= 3:
        raise ValueError("Demo rounds must be 1 to 3")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"dp-fl-{uuid.uuid4().hex[:10]}"
    checkpoint = artifact_dir / f"{run_id}.pt"
    evidence = artifact_dir / f"{run_id}-server.json"
    address = f"127.0.0.1:{_free_port()}"
    ctx = mp.get_context("spawn")
    server = ctx.Process(target=_server, args=(address, rounds, str(checkpoint), str(evidence)))
    processes = [server]
    try:
        server.start()
        deadline = time.monotonic() + 90
        while True:
            if not server.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("DP Flower server did not become ready")
            try:
                with socket.create_connection(("127.0.0.1", int(address.split(':')[1])), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        for index in range(3):
            process = ctx.Process(target=_client, args=(index, address, config))
            process.start()
            processes.append(process)
        deadline = time.monotonic() + 180
        while any(p.is_alive() for p in processes):
            if any(p.exitcode not in (None, 0) for p in processes):
                raise RuntimeError("DP federation subprocess failed")
            if time.monotonic() > deadline:
                raise RuntimeError("DP federation timed out")
            time.sleep(0.2)
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError("DP federation subprocess failed")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=5)
    result = json.loads(evidence.read_text(encoding="utf-8"))
    saved_model = _model()
    set_parameters(saved_model, list(torch.load(checkpoint, weights_only=True).values()))
    output = {"run_id": run_id, "workload": "Flower Opacus toy classifier; not PPE detector",
        "status": "completed", "rounds": rounds, "simulated_sites": list(CLIENTS),
        "differential_privacy": True, "secure_aggregation": False, "secure_rng": False,
        "mechanism": "Opacus DP-SGD", "accountant": "RDP", "config": asdict(config),
        "scope": "Experimental example-level DP within each simulated client; not client-level DP or PPE DP; separate runs not composed",
        "raw_training_data_sent": False, "private_training_metrics_sent": False,
        "bytes_per_client_update": sum(a.nbytes for a in _get_parameters(saved_model)), "model_artifact": checkpoint.name,
        "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), **result}
    version = register_version(db_path, checkpoint, round_number=rounds,
        validation_metric=result["accuracy"][str(rounds)], notes=output["scope"],
        model_kind="toy_classifier", clients=CLIENTS, project_root=artifact_dir.parent.parent)
    output["model_version_id"] = version
    with sqlite3.connect(db_path) as conn:
        for record in result["accounting"]:
            r = record["round"]
            conn.execute("INSERT INTO federation_rounds(round_id,round_number,status,metrics_json) VALUES (?,?,?,?)",
                (f"{run_id}-r{r}", r, "completed", json.dumps({**output, "current_round": r,
                    "client_accounting": record["clients"]})))
        for metrics in result["accounting"][-1]["clients"]:
            conn.execute("INSERT INTO privacy_metrics(metric_id,mechanism,epsilon,delta,clipping_norm,noise_multiplier,source) VALUES (?,?,?,?,?,?,?)",
                (f"{run_id}-{metrics['site_id']}", output["mechanism"], metrics["epsilon"], config.delta,
                 config.clipping_norm, config.noise_multiplier, output["scope"]))
    (artifact_dir / f"{run_id}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output
