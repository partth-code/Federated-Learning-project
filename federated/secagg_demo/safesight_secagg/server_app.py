"""Flower ServerApp that only receives SecAgg+ aggregated model updates."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from safesight_secagg.task import evaluate, held_out_data, initial_parameters


app = ServerApp()


def _read_arrays(path: Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return [archive[key].copy() for key in sorted(archive.files)]


def _run_ppe_secagg(grid: Grid, context: Context) -> None:
    """Securely aggregate real, locally trained PPE detection-head weights."""
    initial = _read_arrays(Path(os.environ["SAFESIGHT_PPE_INITIAL_FILE"]))
    expected_rounds = int(context.run_config["num-server-rounds"])
    num_clients = int(context.run_config["num-clients"])

    class CaptureFedAvg(FedAvg):
        def __init__(self, **kwargs):
            self.secagg_aggregate = None
            super().__init__(**kwargs)

        def aggregate_fit(self, server_round, results, failures):
            aggregated = super().aggregate_fit(server_round, results, failures)
            self.secagg_aggregate = aggregated[0]
            return aggregated

    strategy = CaptureFedAvg(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        fraction_evaluate=0.0,
        min_evaluate_clients=0,
        initial_parameters=ndarrays_to_parameters(initial),
    )
    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=expected_rounds),
        strategy=strategy,
    )
    workflow = DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflow(
            num_shares=int(context.run_config["num-shares"]),
            reconstruction_threshold=int(context.run_config["reconstruction-threshold"]),
            max_weight=float(context.run_config["max-weight"]),
            clipping_range=float(context.run_config["clipping-range"]),
            quantization_range=int(context.run_config["quantization-range"]),
            modulus_range=int(context.run_config["modulus-range"]),
            timeout=float(context.run_config["timeout"]),
        )
    )
    workflow(grid, legacy_context)
    if strategy.secagg_aggregate is None:
        raise RuntimeError("Flower SecAgg+ returned no aggregate for the PPE detection head")
    arrays = parameters_to_ndarrays(strategy.secagg_aggregate)
    if len(arrays) != len(initial) or any(
        current.shape != expected.shape or not np.isfinite(current).all()
        for current, expected in zip(arrays, initial, strict=True)
    ):
        raise RuntimeError("Flower SecAgg+ aggregate failed PPE head-schema validation")
    # Flower's SecAgg+ quantize/dequantize path may promote tensors to float64.
    # Preserve the model's original array dtypes at the checkpoint boundary.
    arrays = [current.astype(expected.dtype, copy=False) for current, expected in zip(arrays, initial, strict=True)]
    output_path = Path(os.environ["SAFESIGHT_PPE_OUTPUT_FILE"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **{f"array_{index:04d}": value for index, value in enumerate(arrays)},
    )
    print(json.dumps({
        "workload": "PPE detection head",
        "secure_aggregation": True,
        "rounds": expected_rounds,
        "clients": num_clients,
        "clipping_range": float(context.run_config["clipping-range"]),
    }))


def persist_run(root: Path, run_id: str, rounds: list[dict[str, float]], expected_rounds: int) -> dict[str, object]:
    completed = [record for record in rounds if record.get("round", -1) > 0]
    if len(completed) != expected_rounds:
        raise RuntimeError(f"SecAgg+ workflow returned {len(completed)} evaluated rounds; expected {expected_rounds}")
    output = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workload": "NumPy linear classifier with deterministic synthetic non-IID data; not PPE detector",
        "protocol": "Flower SecAgg+ (`SecAggPlusWorkflow` + `secaggplus_mod`)",
        "secure_aggregation": True,
        "differential_privacy": False,
        "simulated_sites": ["factory-a", "factory-b", "factory-c"],
        "rounds": completed,
        "server_visibility": "Server strategy receives the aggregate after SecAgg+ unmasking; individual client updates are masked by the protocol.",
    }
    artifacts = root / "artifacts" / "federated"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / f"{run_id}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    db_path = root / "data" / "safesight.sqlite3"
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            for record in completed:
                number = int(record["round"])
                conn.execute(
                    "INSERT OR REPLACE INTO federation_rounds(round_id, round_number, status, metrics_json) VALUES (?, ?, ?, ?)",
                    (f"{run_id}-r{number}", number, "completed", json.dumps({**record, "secure_aggregation": True, "workload": output["workload"]})),
                )
    return output


@app.main()
def main(grid: Grid, context: Context) -> None:
    if os.environ.get("SAFESIGHT_PPE_SECAGG") == "1":
        _run_ppe_secagg(grid, context)
        return
    expected_rounds = int(context.run_config["num-server-rounds"])
    initial = initial_parameters()
    validation_x, validation_y = held_out_data()
    evaluated: list[dict[str, float]] = []

    def evaluate_global(server_round, parameters, config):
        del config
        values = parameters
        val_loss, val_accuracy = evaluate(values, validation_x, validation_y)
        evaluated.append({"round": float(server_round), "loss": val_loss, "accuracy": val_accuracy})
        return val_loss, {"accuracy": val_accuracy}

    num_clients = int(context.run_config["num-clients"])
    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        fraction_evaluate=0.0,
        min_evaluate_clients=0,
        initial_parameters=ndarrays_to_parameters(initial),
        evaluate_fn=evaluate_global,
    )
    legacy_context = LegacyContext(context=context, config=ServerConfig(num_rounds=expected_rounds), strategy=strategy)
    workflow = DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflow(
            num_shares=int(context.run_config["num-shares"]),
            reconstruction_threshold=int(context.run_config["reconstruction-threshold"]),
            max_weight=float(context.run_config["max-weight"]),
            clipping_range=float(context.run_config["clipping-range"]),
            quantization_range=int(context.run_config["quantization-range"]),
            modulus_range=int(context.run_config["modulus-range"]),
            timeout=float(context.run_config["timeout"]),
        )
    )
    workflow(grid, legacy_context)
    root = Path(os.environ.get("SAFESIGHT_PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()
    run_id = f"secagg-{uuid.uuid4().hex[:10]}"
    persist_run(root, run_id, evaluated, expected_rounds)
    print(json.dumps({"run_id": run_id, "secure_aggregation": True, "evaluated_rounds": len(evaluated) - 1}))
