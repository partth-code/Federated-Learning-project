"""One-round PPE detection-head federation using Flower SecAgg+ in WSL."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import shlex
import sqlite3
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config import Settings
from detector.ultralytics_adapter import load_ppe_model
from federated.lightweight_fl import CLIENTS
from federated.ppe_federation import (
    BATCH_SIZE,
    DEMO_EPOCHS,
    DEMO_ROUNDS,
    IMAGE_SIZE,
    _client_process,
    _dataset_root,
    _file_sha256,
    _parameter_delta_l2,
    _sample_records,
    _test_metrics,
    _write_data_yaml,
)
from federated.vision_adapter import PPEHeadAdapter
from models.registry import register_version


NUM_SHARES = 3
RECONSTRUCTION_THRESHOLD = 2
MAX_WEIGHT = 1000.0
QUANTIZATION_RANGE = 4_194_304
MODULUS_RANGE = 4_294_967_296
FLOWER_TIMEOUT_SECONDS = 300.0
MAX_CLIPPING_RANGE = 128.0


def _write_arrays(path: Path, arrays: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{f"array_{index:04d}": value for index, value in enumerate(arrays)},
    )


def _read_arrays(path: Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return [archive[key].copy() for key in sorted(archive.files)]


def _clipping_range(arrays: list[list[np.ndarray]]) -> float:
    """Choose a SecAgg+ bound that contains every transmitted model value."""
    maximum = max(float(np.max(np.abs(array.astype(np.float64)))) for group in arrays for array in group)
    required = math.ceil(maximum * 1.1 * 100.0) / 100.0
    clipping = max(8.0, required)
    if not math.isfinite(clipping) or clipping > MAX_CLIPPING_RANGE:
        raise ValueError(
            f"PPE head value magnitude {maximum:.6g} exceeds the supported SecAgg+ clipping bound"
        )
    return clipping


def _wsl_path(path: Path) -> str:
    return subprocess.check_output(
        ["wsl.exe", "-e", "wslpath", "-a", str(path.resolve())],
        text=True,
    ).strip()


def _run_secagg_flower_app(
    project_root: Path,
    update_dir: Path,
    initial_file: Path,
    counts_file: Path,
    output_file: Path,
    clipping_range: float,
) -> dict[str, Any]:
    app_source = project_root / "federated" / "secagg_demo"
    if not app_source.is_dir():
        raise FileNotFoundError("The local Flower SecAgg+ app is unavailable")
    wsl_app = _wsl_path(app_source)
    wsl_updates = _wsl_path(update_dir)
    wsl_initial = _wsl_path(initial_file)
    wsl_counts = _wsl_path(counts_file)
    wsl_output = _wsl_path(output_file)
    wsl_project = _wsl_path(project_root)
    run_config = (
        "num-server-rounds=1 num-clients=3 num-shares=3 "
        f"reconstruction-threshold=2 max-weight=1000.0 clipping-range={clipping_range:.2f} "
        "quantization-range=4194304 modulus-range=4294967296 timeout=300.0"
    )
    command = (
        "set -euo pipefail; "
        "tmp=$(mktemp -d /tmp/safesight-ppe-secagg.XXXXXX); "
        "cleanup() { "
        "case \"$tmp\" in /tmp/safesight-ppe-secagg.*) ;; *) exit 91 ;; esac; "
        "for proc in /proc/[0-9]*; do "
        "[ -r \"$proc/cmdline\" ] || continue; "
        "cmdline=$(tr '\\0' ' ' < \"$proc/cmdline\" 2>/dev/null || true); "
        "case \"$cmdline\" in \"$tmp/.venv/bin/python $tmp/.venv/bin/flower-superlink \"*) "
        "kill -TERM \"${proc##*/}\" 2>/dev/null || true ;; esac; "
        "done; sleep 1; rm -rf -- \"$tmp\"; "
        "}; trap 'status=$?; trap - EXIT; cleanup; exit \"$status\"' EXIT; "
        f"tar -C {shlex.quote(wsl_app)} --exclude=.venv -cf - . | tar -C \"$tmp\" -xf -; "
        "cd \"$tmp\"; "
        "export SAFESIGHT_PPE_SECAGG=1; "
        f"export SAFESIGHT_PPE_UPDATE_DIR={shlex.quote(wsl_updates)}; "
        f"export SAFESIGHT_PPE_INITIAL_FILE={shlex.quote(wsl_initial)}; "
        f"export SAFESIGHT_PPE_COUNTS_FILE={shlex.quote(wsl_counts)}; "
        f"export SAFESIGHT_PPE_OUTPUT_FILE={shlex.quote(wsl_output)}; "
        f"export SAFESIGHT_PROJECT_ROOT={shlex.quote(wsl_project)}; "
        "uv run --project . flwr run . --stream "
        f"--run-config {shlex.quote(run_config)}"
    )
    completed = subprocess.run(["wsl.exe", "-e", "bash", "-lc", command], cwd=app_source, check=False)
    aggregate_written = output_file.is_file()
    if completed.returncode != 0 and not aggregate_written:
        raise RuntimeError(f"Flower SecAgg+ PPE run failed in WSL (exit={completed.returncode})")
    return {
        "launcher_exit_code": completed.returncode,
        "server_aggregate_file_written": aggregate_written,
    }


def _train_client_updates(
    selected: dict[str, list[dict[str, Any]]],
    client_configs: dict[str, Path],
    work_dir: Path,
    checkpoint: Path,
    schema_sha256: str,
) -> dict[str, Path]:
    output_paths = {site: work_dir / "client-updates" / f"{site}.npz" for site in CLIENTS}
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_client_process,
            args=(
                "",
                site,
                str(checkpoint),
                str(client_configs[site]),
                str(work_dir / "client-runs"),
                len(selected[site]),
                schema_sha256,
                str(output_paths[site]),
            ),
        )
        for site in CLIENTS
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=600)
            if process.is_alive():
                raise TimeoutError("A local PPE client exceeded the 10-minute training safety timeout")
            if process.exitcode != 0:
                raise RuntimeError(f"A local PPE client exited with code {process.exitcode}")
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.is_alive():
                process.join(timeout=5)
        raise
    if any(not path.is_file() for path in output_paths.values()):
        raise RuntimeError("A simulated site did not produce its locally trained PPE head")
    return output_paths


def run_ppe_head_secagg(
    settings: Settings,
    *,
    fraction_per_site: float = 0.05,
    seed: int = 20261009,
) -> dict[str, Any]:
    """Train three local PPE heads, then aggregate them only through Flower SecAgg+."""
    if not np.isfinite(fraction_per_site) or not 0 < fraction_per_site <= 0.05:
        raise ValueError("fraction_per_site must be in (0, 0.05]")
    root = settings.project_root
    manifest = json.loads((root / "artifacts" / "dataset_manifest.json").read_text(encoding="utf-8"))
    split_metadata = json.loads((root / "artifacts" / "dataset_partitions.json").read_text(encoding="utf-8"))
    if split_metadata.get("strategy") != "dominant-class Dirichlet allocation":
        raise RuntimeError("Regenerate the deterministic class-skewed site partitions before PPE SecAgg+")
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

    model = load_ppe_model(checkpoint)
    adapter = PPEHeadAdapter(model)
    initial_parameters = adapter.get_parameters()
    run_id = f"ppe-secagg-{uuid.uuid4().hex[:10]}"
    artifact_dir = root / "artifacts" / "federated"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"{run_id}.pt"

    with tempfile.TemporaryDirectory(prefix=f"safesight-{run_id}-") as temporary:
        work_dir = Path(temporary)
        full_yaml = _write_data_yaml(work_dir / "data", dataset_root, manifest["classes"])
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
        initial_metrics = _test_metrics(model, full_yaml, work_dir / "evaluation", "before_secagg")
        initial_file = work_dir / "initial-head.npz"
        _write_arrays(initial_file, initial_parameters)
        counts_file = work_dir / "site-counts.json"
        site_train_images = {site: len(selected[site]) for site in CLIENTS}
        counts_file.write_text(json.dumps(site_train_images), encoding="utf-8")
        update_paths = _train_client_updates(
            selected,
            client_configs,
            work_dir,
            checkpoint,
            adapter.schema_sha256,
        )
        local_parameters_by_site: dict[str, list[np.ndarray]] = {}
        client_update_checks: dict[str, dict[str, bool]] = {}
        for site, path in update_paths.items():
            local_parameters = _read_arrays(path)
            if len(local_parameters) != len(initial_parameters) or _parameter_delta_l2(initial_parameters, local_parameters) <= 0:
                raise RuntimeError(f"{site} did not produce a changed PPE head before SecAgg+")
            if any(
                current.shape != initial.shape
                or current.dtype != initial.dtype
                or not np.isfinite(current).all()
                for current, initial in zip(local_parameters, initial_parameters, strict=True)
            ):
                raise RuntimeError(f"{site} PPE head failed schema validation before SecAgg+")
            local_parameters_by_site[site] = local_parameters
            client_update_checks[site] = {
                "local_training_completed": True,
                "nonzero_head_change": True,
            }
        clipping_range = _clipping_range([initial_parameters, *local_parameters_by_site.values()])
        global_file = work_dir / "secagg-global-head.npz"
        flower_completion = _run_secagg_flower_app(
            root,
            work_dir / "client-updates",
            initial_file,
            counts_file,
            global_file,
            clipping_range,
        )
        if not global_file.is_file():
            raise RuntimeError("Flower SecAgg+ completed without producing an aggregate PPE head")
        global_parameters = _read_arrays(global_file)
        if len(global_parameters) != len(initial_parameters):
            raise RuntimeError("Flower SecAgg+ aggregate has an incompatible PPE head schema")
        global_parameters = [value.astype(expected.dtype, copy=False) for value, expected in zip(global_parameters, initial_parameters, strict=True)]
        if any(
            current.shape != expected.shape or not np.isfinite(current).all()
            for current, expected in zip(global_parameters, initial_parameters, strict=True)
        ):
            raise RuntimeError("Flower SecAgg+ aggregate failed numeric/schema validation")
        global_delta_l2 = _parameter_delta_l2(initial_parameters, global_parameters)
        if global_delta_l2 <= 0:
            raise RuntimeError("Flower SecAgg+ returned an unchanged PPE detection head")
        adapter.set_parameters(global_parameters)
        model.save(model_path)
        reloaded = load_ppe_model(model_path)
        if PPEHeadAdapter(reloaded).schema_sha256 != adapter.schema_sha256:
            raise RuntimeError("SecAgg+ global PPE checkpoint failed schema verification")
        final_metrics = _test_metrics(reloaded, full_yaml, work_dir / "evaluation", "after_secagg")

    version_id = register_version(
        settings.db_path,
        model_path,
        round_number=1,
        validation_metric=None,
        notes="One-round Flower SecAgg+ PPE detection-head demo; one local epoch/client; DP disabled.",
        model_kind="ppe_detector_head_only",
        clients=CLIENTS,
        project_root=root,
    )
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
        "protocol": "Flower SecAggPlusWorkflow + secaggplus_mod; one secure aggregation round",
        "flower_completion": flower_completion,
        "simulated_sites": list(CLIENTS),
        "site_train_images": site_train_images,
        "client_update_checks": client_update_checks,
        "site_class_instances": site_class_instances,
        "partition_strategy": split_metadata["strategy"],
        "partition_seed": split_metadata["seed"],
        "fraction_per_site": fraction_per_site,
        "local_epochs": DEMO_EPOCHS,
        "rounds": DEMO_ROUNDS,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "nominal_batch_size": BATCH_SIZE,
        "device": "cpu",
        "trainable_head_parameter_count": adapter.trainable_parameter_count,
        "bytes_per_client_update": adapter.bytes_per_update,
        "weight_updates_changed": True,
        "global_head_delta_l2": global_delta_l2,
        "global_model_sha256": _file_sha256(model_path),
        "schema_sha256": adapter.schema_sha256,
        "initial_test_metrics": initial_metrics,
        "post_federation_test_metrics": final_metrics,
        "model_artifact": model_path.name,
        "model_version_id": version_id,
        "secure_aggregation": True,
        "differential_privacy": False,
        "raw_training_images_sent": False,
        "secagg_parameters": {
            "num_shares": NUM_SHARES,
            "reconstruction_threshold": RECONSTRUCTION_THRESHOLD,
            "max_weight": MAX_WEIGHT,
            "clipping_range": clipping_range,
            "quantization_range": QUANTIZATION_RANGE,
            "modulus_range": MODULUS_RANGE,
        },
        "server_visibility": (
            "The Flower ServerApp received the SecAgg+ aggregate, not each individual head update. "
            "Factory A/B/C and the temporary client-update files were simulated on one laptop; "
            "a same-machine administrator can inspect local processes/files."
        ),
    }
    artifact_path = artifact_dir / f"{run_id}.json"
    artifact_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    (artifact_dir / "ppe_head_fl_latest.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    metrics = {
        key: output[key]
        for key in (
            "workload",
            "protocol",
            "flower_completion",
            "simulated_sites",
            "site_train_images",
            "client_update_checks",
            "bytes_per_client_update",
            "initial_test_metrics",
            "post_federation_test_metrics",
            "weight_updates_changed",
            "global_head_delta_l2",
            "global_model_sha256",
            "secure_aggregation",
            "differential_privacy",
            "raw_training_images_sent",
        )
    }
    with sqlite3.connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO federation_rounds(round_id, round_number, status, metrics_json) VALUES (?, ?, ?, ?)",
            (f"{run_id}-r1", 1, "completed", json.dumps(metrics)),
        )
        conn.commit()
    return output
