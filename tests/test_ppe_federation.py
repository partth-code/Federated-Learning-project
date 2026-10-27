from __future__ import annotations

import pytest
import numpy as np
import hashlib

from federated.lightweight_fl import CLIENTS
from federated.ppe_federation import _aggregate_fit_metrics, _parameter_delta_l2
from federated.ppe_secagg import _clipping_range
from federated.verification import is_verified_ppe_federation_run


def _client_fit_metrics(site_id: str, delta: float = 0.5):
    return {
        "site_id": site_id,
        "schema_sha256": "a" * 64,
        "local_train_images": 5,
        "local_epochs": 1.0,
        "local_head_delta_l2": delta,
    }


def test_fit_metrics_require_a_nonzero_update_from_each_simulated_client():
    results = [(5, _client_fit_metrics(site_id)) for site_id in CLIENTS]
    aggregated = _aggregate_fit_metrics(results)
    assert aggregated["factory-a_head_delta_l2"] == 0.5
    assert aggregated["factory-c_train_images"] == 5

    with pytest.raises(RuntimeError, match="exactly one"):
        _aggregate_fit_metrics(results[:2])

    no_update = [(5, _client_fit_metrics(site_id, 0.0)) for site_id in CLIENTS]
    with pytest.raises(RuntimeError, match="nonzero"):
        _aggregate_fit_metrics(no_update)


def test_aggregate_head_delta_requires_matching_nonempty_schemas():
    assert _parameter_delta_l2([np.array([1.0])], [np.array([2.0])]) == 1.0
    with pytest.raises(ValueError, match="same nonzero length"):
        _parameter_delta_l2([], [])


def test_secagg_clipping_bound_contains_actual_model_tensors_without_silent_clipping():
    bound = _clipping_range(
        [
            [np.array([-1.0, 2.0], dtype=np.float32)],
            [np.array([-9.55, 0.25], dtype=np.float32)],
        ]
    )
    assert bound == 10.51
    assert _clipping_range([[np.array([1.0], dtype=np.float32)]]) == 8.0
    with pytest.raises(ValueError, match="exceeds"):
        _clipping_range([[np.array([129.0], dtype=np.float32)]])


def test_dashboard_verification_requires_real_client_updates_and_matching_checkpoint_hash(tmp_path):
    checkpoint = tmp_path / "global.pt"
    checkpoint.write_bytes(b"verified model bytes")
    metrics = {}
    sites = ["factory-a", "factory-b", "factory-c"]
    for site in sites:
        metrics[f"{site}_head_delta_l2"] = 0.25
        metrics[f"{site}_train_images"] = 3
    run = {
        "status": "completed",
        "weight_updates_changed": True,
        "model_artifact": checkpoint.name,
        "global_model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "global_head_delta_l2": 0.5,
        "simulated_sites": sites,
        "client_fit_metrics": metrics,
    }
    assert is_verified_ppe_federation_run(tmp_path, run)
    secure_run = {
        **run,
        "secure_aggregation": True,
        "protocol": "Flower SecAggPlusWorkflow + secaggplus_mod; one secure aggregation round",
        "flower_completion": {
            "launcher_exit_code": 1,
            "server_aggregate_file_written": True,
        },
        "raw_training_images_sent": False,
        "site_train_images": {site: 5 for site in sites},
        "client_update_checks": {
            site: {"local_training_completed": True, "nonzero_head_change": True}
            for site in sites
        },
    }
    secure_run.pop("client_fit_metrics")
    assert is_verified_ppe_federation_run(tmp_path, secure_run)
    assert not is_verified_ppe_federation_run(
        tmp_path, {**secure_run, "protocol": "unspecified secure protocol"}
    )
    checkpoint.write_bytes(b"tampered model bytes")
    assert not is_verified_ppe_federation_run(tmp_path, run)
