"""Artifact integrity checks for recorded PPE federation runs."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any


def is_verified_ppe_federation_run(artifact_dir: Path, run: dict[str, Any] | None) -> bool:
    if not isinstance(run, dict) or run.get("status") != "completed" or run.get("weight_updates_changed") is not True:
        return False
    model_name = run.get("model_artifact")
    expected_hash = run.get("global_model_sha256")
    if not isinstance(model_name, str) or Path(model_name).name != model_name:
        return False
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        return False
    try:
        int(expected_hash, 16)
        global_delta = float(run.get("global_head_delta_l2", 0))
    except (TypeError, ValueError):
        return False
    if not math.isfinite(global_delta) or global_delta <= 0:
        return False

    sites = run.get("simulated_sites")
    client_metrics = run.get("client_fit_metrics")
    if (
        not isinstance(sites, list)
        or len(sites) != 3
        or any(not isinstance(site, str) for site in sites)
        or len(set(sites)) != 3
    ):
        return False
    if run.get("secure_aggregation") is True:
        protocol = run.get("protocol")
        completion = run.get("flower_completion")
        counts = run.get("site_train_images")
        checks = run.get("client_update_checks")
        if (
            not isinstance(protocol, str)
            or "SecAggPlusWorkflow" not in protocol
            or "secaggplus_mod" not in protocol
            or not isinstance(completion, dict)
            or completion.get("server_aggregate_file_written") is not True
            or not isinstance(completion.get("launcher_exit_code"), int)
            or completion["launcher_exit_code"] not in (0, 1)
            or run.get("raw_training_images_sent") is not False
            or not isinstance(counts, dict)
            or set(counts) != set(sites)
            or not isinstance(checks, dict)
            or set(checks) != set(sites)
        ):
            return False
        for site in sites:
            try:
                examples = int(counts[site])
            except (TypeError, ValueError):
                return False
            if examples < 1 or not isinstance(checks[site], dict):
                return False
            if checks[site].get("local_training_completed") is not True:
                return False
            if checks[site].get("nonzero_head_change") is not True:
                return False
    else:
        if not isinstance(client_metrics, dict):
            return False
        for site in sites:
            try:
                changed = float(client_metrics[f"{site}_head_delta_l2"])
                examples = int(client_metrics[f"{site}_train_images"])
            except (KeyError, TypeError, ValueError):
                return False
            if not math.isfinite(changed) or changed <= 0 or examples < 1:
                return False

    model_path = artifact_dir / model_name
    if not model_path.is_file():
        return False
    digest = hashlib.sha256()
    with model_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_hash
