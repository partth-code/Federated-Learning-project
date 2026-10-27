"""One-round, CPU-only Flower experiment for a frozen PPE detection head."""

from __future__ import annotations

import json
import hashlib
import math
import multiprocessing as mp
import socket
import sqlite3
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from flwr.client import start_client
from flwr.common import Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig, start_server

from config import Settings
from detector.ultralytics_adapter import load_ppe_model
from federated.lightweight_fl import CLIENTS
from federated.vision_adapter import PPEHeadAdapter, PPEHeadFedAvg, PPEHeadFlowerClient
from models.registry import register_version


MAX_DEMO_FRACTION = 0.05
DEMO_EPOCHS = 1
DEMO_ROUNDS = 1
IMAGE_SIZE = 320
BATCH_SIZE = 4


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(address: str, process: mp.Process, timeout: float = 180.0) -> None:
    host, port_text = address.rsplit(":", 1)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process.is_alive():
            raise RuntimeError(f"Flower server exited before binding its local endpoint (exit={process.exitcode})")
        try:
            with socket.create_connection((host, int(port_text)), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError("Flower server did not bind its local endpoint within 180 seconds")


def _stop_processes(processes: list[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        if process.is_alive():
            process.join(timeout=5)


def _dataset_root(settings: Settings) -> Path:
    if settings.dataset_root is not None:
        root = settings.dataset_root
    else:
        from ultralytics import settings as ultralytics_settings

        root = Path(ultralytics_settings["datasets_dir"]) / "construction-ppe"
    root = root.resolve()
    if not (root / "images" / "train").is_dir():
        raise FileNotFoundError("Local Construction-PPE training images are unavailable; no download was attempted")
    return root


def _write_data_yaml(
    work_dir: Path,
    dataset_root: Path,
    class_names: dict[str, str],
    *,
    site_records: list[dict[str, Any]] | None = None,
    site_id: str | None = None,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "path": dataset_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(class_id): label for class_id, label in class_names.items()},
    }
    if site_records is not None:
        if not site_id:
            raise ValueError("site_id is required when creating a client-specific data file")
        image_list = work_dir / f"{site_id}-train.txt"
        images = [(dataset_root / Path(str(record["image"]))).resolve() for record in site_records]
        if not images or any(not image.is_file() for image in images):
            raise FileNotFoundError(f"A local image is missing from the {site_id} simulated partition")
        image_list.write_text("\n".join(path.as_posix() for path in images), encoding="utf-8")
        data["train"] = image_list.as_posix()
    data_path = work_dir / (f"{site_id}.yaml" if site_id else "dataset.yaml")
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_path


def _test_metrics(model: Any, data_yaml: Path, output_dir: Path, name: str) -> dict[str, float]:
    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        workers=0,
        plots=False,
        project=str(output_dir),
        name=name,
        exist_ok=True,
        verbose=False,
    )
    box = getattr(results, "box", None)
    if box is None:
        raise RuntimeError("Ultralytics did not return held-out detection metrics")
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    }


def _aggregate_fit_metrics(
    client_metrics: list[tuple[int, dict[str, Scalar]]],
) -> dict[str, Scalar]:
    if len(client_metrics) != len(CLIENTS):
        raise RuntimeError("Expected exactly one PPE head update from each simulated client")
    by_site = {str(metrics.get("site_id")): metrics for _, metrics in client_metrics}
    if set(by_site) != set(CLIENTS):
        raise RuntimeError("Expected one reported PPE head update from each simulated client")
    aggregated: dict[str, Scalar] = {}
    for site_id in CLIENTS:
        metrics = by_site[site_id]
        delta = float(metrics.get("local_head_delta_l2", 0.0))
        if not math.isfinite(delta) or delta <= 0:
            raise RuntimeError(f"{site_id} reported no finite, nonzero PPE head update")
        aggregated[f"{site_id}_train_images"] = int(metrics["local_train_images"])
        aggregated[f"{site_id}_local_epochs"] = float(metrics["local_epochs"])
        aggregated[f"{site_id}_head_delta_l2"] = delta
    return aggregated


def _parameter_delta_l2(before: list[np.ndarray], after: list[np.ndarray]) -> float:
    if len(before) != len(after) or not before:
        raise ValueError("PPE head parameter lists must have the same nonzero length")
    return float(
        np.sqrt(
            sum(
                np.sum((new.astype(np.float64) - old.astype(np.float64)) ** 2)
                for old, new in zip(before, after, strict=True)
            )
        )
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _server_process(
    address: str,
    checkpoint_path: str,
    full_data_yaml: str,
    temporary_dir: str,
    global_model_path: str,
    expected_clients: int,
    result_queue: Any,
) -> None:
    torch.set_num_threads(1)
    try:
        model = load_ppe_model(Path(checkpoint_path))
        adapter = PPEHeadAdapter(model)
        initial_parameters = adapter.get_parameters()
        temp = Path(temporary_dir)
        initial_test = _test_metrics(model, Path(full_data_yaml), temp / "evaluation", "before")
        strategy = PPEHeadFedAvg(
            expected_schema_sha256=adapter.schema_sha256,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=expected_clients,
            min_evaluate_clients=0,
            min_available_clients=expected_clients,
            initial_parameters=ndarrays_to_parameters(adapter.get_parameters()),
            fit_metrics_aggregation_fn=_aggregate_fit_metrics,
        )
        start_server(
            server_address=address,
            config=ServerConfig(num_rounds=DEMO_ROUNDS),
            strategy=strategy,
        )
        if strategy.last_aggregated_parameters is None:
            raise RuntimeError("Flower completed without an aggregated PPE head")
        global_parameters = parameters_to_ndarrays(strategy.last_aggregated_parameters)
        global_delta_l2 = _parameter_delta_l2(initial_parameters, global_parameters)
        if global_delta_l2 <= 0:
            raise RuntimeError("Flower aggregation returned an unchanged PPE detection head")
        if strategy.last_aggregated_fit_metrics is None:
            raise RuntimeError("Flower did not return verified per-client PPE update metrics")
        adapter.set_parameters(global_parameters)
        model_path = Path(global_model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        reloaded = load_ppe_model(model_path)
        reloaded_adapter = PPEHeadAdapter(reloaded)
        if reloaded_adapter.schema_sha256 != adapter.schema_sha256:
            raise RuntimeError("Saved global PPE artifact failed schema verification")
        final_test = _test_metrics(reloaded, Path(full_data_yaml), temp / "evaluation", "after")
        result_queue.put(
            {
                "global_model_artifact": model_path.name,
                "initial_test_metrics": initial_test,
                "post_federation_test_metrics": final_test,
                "schema_sha256": adapter.schema_sha256,
                "trainable_head_parameter_count": adapter.trainable_parameter_count,
                "bytes_per_client_update": adapter.bytes_per_update,
                "client_fit_metrics": strategy.last_aggregated_fit_metrics,
                "global_head_delta_l2": global_delta_l2,
                "global_model_sha256": _file_sha256(model_path),
            }
        )
    except BaseException as exc:
        result_queue.put({"error": f"{type(exc).__name__}: {exc}"})
        raise


def _client_process(
    address: str,
    site_id: str,
    checkpoint_path: str,
    data_yaml: str,
    local_run_dir: str,
    num_examples: int,
    expected_schema_sha256: str,
    update_output_path: str | None = None,
) -> None:
    torch.set_num_threads(1)
    model = load_ppe_model(Path(checkpoint_path))
    adapter = PPEHeadAdapter(model)
    if adapter.schema_sha256 != expected_schema_sha256:
        raise RuntimeError(f"{site_id} loaded an incompatible PPE checkpoint schema")

    def train_local(core: torch.nn.Module, config: dict[str, Scalar]) -> dict[str, Scalar]:
        del config
        if core is not model.model:
            raise RuntimeError("Flower client model reference changed before local training")
        frozen_before = adapter.backbone_snapshot()
        head_before = adapter.get_parameters()
        model.train(
            data=data_yaml,
            epochs=DEMO_EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            nbs=BATCH_SIZE,
            device="cpu",
            workers=0,
            freeze=len(model.model.model) - 1,
            fraction=1.0,
            optimizer="AdamW",
            lr0=0.0005,
            warmup_epochs=0.0,
            mosaic=0.0,
            close_mosaic=0,
            val=False,
            save=True,
            plots=False,
            amp=False,
            cache=False,
            project=local_run_dir,
            name=site_id,
            exist_ok=True,
            verbose=False,
        )
        trained_adapter = PPEHeadAdapter(model)
        trained_adapter.assert_backbone_unchanged(frozen_before)
        adapter.rebind(model)
        head_after = adapter.get_parameters()
        if all(np.array_equal(before, after) for before, after in zip(head_before, head_after, strict=True)):
            raise RuntimeError(f"{site_id} local training returned an unchanged PPE detection head")
        delta_l2 = float(
            np.sqrt(sum(np.sum((after.astype(np.float64) - before.astype(np.float64)) ** 2)
                        for before, after in zip(head_before, head_after, strict=True)))
        )
        return {
            "local_train_images": int(num_examples),
            "local_epochs": float(DEMO_EPOCHS),
            "local_head_delta_l2": delta_l2,
        }

    client = PPEHeadFlowerClient(
        adapter,
        site_id=site_id,
        num_examples=num_examples,
        train_local=train_local,
        evaluate_local=None,
    )
    if update_output_path is not None:
        adapter.freeze_backbone()
        train_local(adapter.model, {})
        destination = Path(update_output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            **{f"array_{index:04d}": value for index, value in enumerate(adapter.get_parameters())},
        )
        return
    start_client(server_address=address, client=client.to_client())


def _sample_records(
    partition: list[dict[str, Any]], fraction: float, seed: int, site_id: str
) -> list[dict[str, Any]]:
    sample_count = max(1, math.ceil(len(partition) * fraction))
    if sample_count > len(partition):
        raise ValueError(f"The {site_id} partition has no train images")
    site_seed = seed + sum((index + 1) * ord(char) for index, char in enumerate(site_id))
    rng = np.random.default_rng(site_seed)
    chosen = sorted(int(index) for index in rng.choice(len(partition), size=sample_count, replace=False))
    return [partition[index] for index in chosen]


def run_ppe_head_federation(
    settings: Settings,
    *,
    fraction_per_site: float = MAX_DEMO_FRACTION,
    seed: int = 20261009,
) -> dict[str, Any]:
    """Run one short Flower FedAvg round; never expands into full training."""
    if not np.isfinite(fraction_per_site) or not 0 < fraction_per_site <= MAX_DEMO_FRACTION:
        raise ValueError(f"fraction_per_site must be in (0, {MAX_DEMO_FRACTION}]")

    root = settings.project_root
    manifest = json.loads((root / "artifacts" / "dataset_manifest.json").read_text(encoding="utf-8"))
    split_metadata = json.loads((root / "artifacts" / "dataset_partitions.json").read_text(encoding="utf-8"))
    if split_metadata.get("strategy") != "dominant-class Dirichlet allocation":
        raise RuntimeError("Regenerate the deterministic class-skewed client partitions before PPE federation")
    partitions = split_metadata["partitions"]
    if set(partitions) != set(CLIENTS):
        raise RuntimeError("Expected the three simulated Factory A/B/C client partitions")
    dataset_root = _dataset_root(settings)
    checkpoint = root / "artifacts" / "runs" / "construction_ppe_demo" / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("The local demo PPE checkpoint is unavailable; no replacement was downloaded")

    selected = {
        site: _sample_records(partitions[site], fraction_per_site, seed, site)
        for site in CLIENTS
    }
    for records in selected.values():
        if any(not (dataset_root / Path(str(record["image"]))).is_file() for record in records):
            raise FileNotFoundError("A selected image in the simulated client manifest is unavailable")

    initial_model = load_ppe_model(checkpoint)
    initial_adapter = PPEHeadAdapter(initial_model)
    run_id = f"ppe-fl-{uuid.uuid4().hex[:10]}"
    artifact_dir = root / "artifacts" / "federated"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"{run_id}.pt"

    with tempfile.TemporaryDirectory(prefix=f"safesight-{run_id}-") as work:
        work_dir = Path(work)
        full_config = _write_data_yaml(work_dir / "data", dataset_root, manifest["classes"])
        client_configs = {
            site: _write_data_yaml(
                work_dir / "data",
                dataset_root,
                manifest["classes"],
                site_records=selected[site],
                site_id=site,
            )
            for site in CLIENTS
        }
        address = f"127.0.0.1:{_free_port()}"
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        server = context.Process(
            target=_server_process,
            args=(
                address,
                str(checkpoint),
                str(full_config),
                str(work_dir),
                str(model_path),
                len(CLIENTS),
                result_queue,
            ),
        )
        clients = [
            context.Process(
                target=_client_process,
                args=(
                    address,
                    site,
                    str(checkpoint),
                    str(client_configs[site]),
                    str(work_dir / "client-runs"),
                    len(selected[site]),
                    initial_adapter.schema_sha256,
                ),
            )
            for site in CLIENTS
        ]
        processes = [server, *clients]
        server.start()
        try:
            _wait_for_server(address, server)
            for process in clients:
                process.start()
            for process in clients:
                process.join(timeout=600)
                if process.is_alive():
                    raise TimeoutError("A PPE Flower client exceeded the 10-minute safety timeout")
                if process.exitcode != 0:
                    raise RuntimeError(f"A PPE Flower client exited with code {process.exitcode}")
            server.join(timeout=600)
            if server.is_alive():
                raise TimeoutError("One-round PPE Flower server exceeded the 10-minute safety timeout")
            if server.exitcode != 0:
                raise RuntimeError(f"PPE Flower server exited with code {server.exitcode}")
            result = result_queue.get(timeout=10)
            if "error" in result:
                raise RuntimeError(f"PPE Flower server reported an error: {result['error']}")
        except BaseException:
            _stop_processes(processes)
            raise
        finally:
            result_queue.close()

    if not model_path.is_file():
        raise RuntimeError("Flower did not produce a verified global PPE model artifact")
    version_id = register_version(
        settings.db_path,
        model_path,
        round_number=DEMO_ROUNDS,
        validation_metric=None,
        notes="One-round frozen-backbone Flower PPE head demo; held-out test metrics are recorded separately.",
        model_kind="ppe_detector_head_only",
        clients=CLIENTS,
        project_root=settings.project_root,
    )
    site_train_images = {site: len(selected[site]) for site in CLIENTS}
    site_class_instances = {
        site: {
            str(class_id): sum(record["class_ids"].count(int(class_id)) for record in selected[site])
            for class_id in manifest["classes"]
        }
        for site in CLIENTS
    }
    output: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "workload": "Ultralytics YOLO11n Construction-PPE detection-head fine-tuning; not full-model federation",
        "protocol": "Flower FedAvg; one local epoch and one aggregation round",
        "simulated_sites": list(CLIENTS),
        "site_train_images": site_train_images,
        "site_class_instances": site_class_instances,
        "partition_strategy": split_metadata["strategy"],
        "partition_seed": split_metadata["seed"],
        "fraction_per_site": fraction_per_site,
        "local_epochs": DEMO_EPOCHS,
        "rounds": DEMO_ROUNDS,
        "image_size": IMAGE_SIZE,
        "device": "cpu",
        "trainable_head_parameter_count": result["trainable_head_parameter_count"],
        "bytes_per_client_update": result["bytes_per_client_update"],
        "client_fit_metrics": result["client_fit_metrics"],
        "weight_updates_changed": True,
        "global_head_delta_l2": result["global_head_delta_l2"],
        "global_model_sha256": result["global_model_sha256"],
        "schema_sha256": result["schema_sha256"],
        "initial_test_metrics": result["initial_test_metrics"],
        "post_federation_test_metrics": result["post_federation_test_metrics"],
        "model_artifact": model_path.name,
        "model_version_id": version_id,
        "raw_training_images_sent": False,
        "secure_aggregation": False,
        "differential_privacy": False,
    }
    artifact_path = artifact_dir / f"{run_id}.json"
    artifact_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    (artifact_dir / "ppe_head_fl_latest.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    metrics = {
        "workload": output["workload"],
        "protocol": output["protocol"],
        "simulated_sites": list(CLIENTS),
        "site_train_images": site_train_images,
        "bytes_per_client_update": output["bytes_per_client_update"],
        "initial_test_metrics": output["initial_test_metrics"],
        "post_federation_test_metrics": output["post_federation_test_metrics"],
        "client_fit_metrics": output["client_fit_metrics"],
        "weight_updates_changed": output["weight_updates_changed"],
        "global_head_delta_l2": output["global_head_delta_l2"],
        "global_model_sha256": output["global_model_sha256"],
        "secure_aggregation": False,
        "differential_privacy": False,
        "raw_training_images_sent": False,
    }
    with sqlite3.connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO federation_rounds(round_id, round_number, status, metrics_json) VALUES (?, ?, ?, ?)",
            (f"{run_id}-r1", 1, "completed", json.dumps(metrics)),
        )
        conn.commit()
    return output
