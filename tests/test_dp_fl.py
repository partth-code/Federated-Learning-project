import numpy as np
import pytest
import torch

from federated.dp_fl import DPClient, DPConfig, set_parameters


def test_cumulative_accounting_and_no_private_metrics():
    torch.set_num_threads(1)
    client = DPClient(0, DPConfig())
    parameters, count, first = client.fit(client.get_parameters({}), {"server_round": 1})
    _, _, second = client.fit(parameters, {"server_round": 2})
    assert count == 80
    assert first["steps"] == 5 and second["steps"] == 10
    assert second["epsilon"] > first["epsilon"] > 0
    assert set(second) == {"site_id", "round", "steps", "epsilon", "delta", "sample_rate"}
    with pytest.raises(RuntimeError):
        client.evaluate(parameters, {})
    with pytest.raises(ValueError):
        client.fit(parameters, {"server_round": 2})


@pytest.mark.parametrize("config", [DPConfig(noise_multiplier=float('nan')),
    DPConfig(clipping_norm=float('inf')), DPConfig(delta=1), DPConfig(epochs=0)])
def test_invalid_config(config):
    with pytest.raises(ValueError):
        config.validate()


def test_parameters_reject_invalid_arrays():
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError):
        set_parameters(model, [])
    with pytest.raises(ValueError):
        set_parameters(model, [np.full((2, 2), np.nan, dtype=np.float32), np.zeros(2, dtype=np.float32)])
    with pytest.raises(ValueError):
        set_parameters(model, [np.zeros((2, 3), dtype=np.float32), np.zeros(2, dtype=np.float32)])


def test_model_initialization_does_not_reset_global_rng():
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    DPClient(1, DPConfig())
    assert torch.equal(before, torch.random.get_rng_state())


def test_status_excludes_baseline_and_failed_dp(tmp_path):
    import json
    import sqlite3
    from database import init_db
    from privacy.status import current_status
    db = init_db(tmp_path / "test.sqlite3")
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO federation_rounds(round_id,round_number,status) VALUES ('baseline',0,'completed')")
        conn.execute("INSERT INTO federation_rounds(round_id,round_number,status,metrics_json) VALUES ('failed',1,'failed',?)",
                     (json.dumps({"differential_privacy": True}),))
    state = current_status(db, True)
    assert state["completed_rounds"] == 0
    assert state["federated_dp"] is None
    assert state["differential_privacy"].startswith("DISABLED")
