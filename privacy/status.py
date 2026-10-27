from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def current_status(db_path: Path, federation_online: bool) -> dict[str, object]:
    with sqlite3.connect(db_path) as conn:
        rounds = int(conn.execute("SELECT COUNT(*) FROM federation_rounds WHERE status = 'completed' AND round_number > 0").fetchone()[0])
        clients = int(conn.execute("SELECT COUNT(DISTINCT site_id) FROM sites WHERE is_simulated = 1").fetchone()[0])
        privacy = conn.execute("SELECT mechanism, epsilon, delta, clipping_norm, noise_multiplier, source FROM privacy_metrics ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
        recorded = conn.execute(
            "SELECT round_id, round_number, created_at, metrics_json FROM federation_rounds "
            "WHERE status = 'completed' AND round_number > 0 ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
        federation_metrics = [
            row[0]
            for row in conn.execute(
                "SELECT metrics_json FROM federation_rounds WHERE status = 'completed' AND round_number > 0 AND metrics_json IS NOT NULL ORDER BY created_at DESC, rowid DESC"
            )
        ]
    parsed_metrics = [json.loads(value) for value in federation_metrics if _valid_json_object(value)]
    dp_run = next((m for m in parsed_metrics if
        m.get("workload") == "Flower Opacus toy classifier; not PPE detector"
        and m.get("status") == "completed" and m.get("differential_privacy") is True
        and _valid_dp(m)), None)
    secagg_verified = any(
        metrics.get("secure_aggregation") is True for metrics in parsed_metrics
    )
    ppe_run = next(
        (
            metrics
            for metrics in parsed_metrics
            if str(metrics.get("workload", "")).startswith("Ultralytics YOLO11n Construction-PPE")
        ),
        None,
    )
    ppe_federation = None
    update_bytes = None
    if ppe_run is not None:
        update_bytes = ppe_run.get("bytes_per_client_update")
        ppe_federation = {
            "scope": "detection head only; experimental local integration",
            "rounds": 1,
            "simulated_clients": len(ppe_run.get("simulated_sites", [])),
            "train_images_per_client": ppe_run.get("site_train_images"),
            "update_bytes_per_client_per_direction": update_bytes,
            "raw_training_images_sent": ppe_run.get("raw_training_images_sent") is True,
            "secure_aggregation": ppe_run.get("secure_aggregation") is True,
            "differential_privacy": ppe_run.get("differential_privacy") is True,
        }
    return {
        "local_processing": True,
        "raw_cctv_federated": False,
        "raw_training_images_federated": False,
        "federation_online": federation_online,
        "completed_rounds": rounds,
        "simulated_clients": clients,
        "secure_aggregation": "ENABLED — verified Flower SecAgg+ integration run" if secagg_verified else "DISABLED — no verified SecAgg run",
        "differential_privacy": "EXPERIMENTAL — Flower + Opacus toy clients; not PPE; secure RNG off" if dp_run else ("AVAILABLE — Opacus toy workload only" if privacy is not None else "DISABLED — no verified DP run"),
        "federated_dp": None if dp_run is None else {
            "run_id": dp_run["run_id"], "scope": dp_run["scope"],
            "secure_rng": dp_run["secure_rng"], "secure_aggregation": dp_run["secure_aggregation"],
            "accountant": dp_run["accountant"], "config": dp_run["config"],
            "accounting": dp_run["accounting"],
        },
        "privacy_metrics": None if privacy is None else {"mechanism": privacy[0], "epsilon": privacy[1], "delta": privacy[2], "clipping_norm": privacy[3], "noise_multiplier": privacy[4], "scope": privacy[5]},
        "recorded_runs": _run_summaries(recorded),
        "status_basis": "Historical completed local experiment records; not live server telemetry or a security attestation",
        "connectivity_basis": "User-controlled demo toggle; not a network health probe",
        "update_bytes": update_bytes,
        "ppe_federation": ppe_federation,
    }


def _valid_json_object(value: str) -> bool:
    try:
        return isinstance(json.loads(value), dict)
    except (json.JSONDecodeError, TypeError):
        return False


def _valid_dp(metrics: dict) -> bool:
    """Malformed partial records must not crash the page or enable its DP panel."""
    import math
    required = ("run_id", "scope", "secure_rng", "secure_aggregation", "accountant", "config")
    if not all(key in metrics for key in required) or not isinstance(metrics.get("accounting"), list):
        return False
    try:
        return bool(metrics["accounting"]) and all(
            type(record["round"]) is int and record["round"] > 0
            and len(record["clients"]) == 3
            and {c["site_id"] for c in record["clients"]} == {"factory-a", "factory-b", "factory-c"}
            and all(math.isfinite(c["epsilon"]) and c["epsilon"] > 0
                    and 0 < c["delta"] < 1 and c["steps"] > 0 for c in record["clients"])
            for record in metrics["accounting"])
    except (KeyError, TypeError, ValueError):
        return False


def _run_summaries(rows) -> list[dict]:
    runs = {}
    for round_id, number, timestamp, raw in rows:
        if not _valid_json_object(raw):
            continue
        metrics = json.loads(raw)
        run_id = metrics.get("run_id") or round_id.rsplit("-r", 1)[0]
        if not isinstance(run_id, str):
            continue
        if run_id in runs:
            runs[run_id]["latest_completed_round"] = max(number, runs[run_id]["latest_completed_round"])
            continue
        size = metrics.get("bytes_per_client_update")
        sites = metrics.get("simulated_sites")
        runs[run_id] = {
            "run_id": run_id, "workload": metrics.get("workload", "Legacy integration record; scope not recorded"),
            "recorded_at": timestamp, "latest_completed_round": number,
            "simulated_clients": len(sites) if isinstance(sites, list) else None,
            "secure_aggregation": metrics.get("secure_aggregation"),
            "differential_privacy": metrics.get("differential_privacy"),
            "raw_update_bytes_per_direction": size if type(size) is int and size >= 0 else None,
        }
    return list(runs.values())
