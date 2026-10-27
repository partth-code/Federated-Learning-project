from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from federated.vision_adapter import PPEHeadAdapter, PPEHeadFedAvg, PPEHeadFlowerClient


class _FakeCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))


class _FakeDetector:
    def __init__(self, class_names=None):
        self.model = _FakeCore()
        self.names = class_names or {0: "helmet", 1: "no_helmet"}


def _callbacks():
    def train(model, config):
        del config
        model.model(torch.ones((2, 3))).sum().backward()
        with torch.no_grad():
            for parameter in model.model[-1].parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-0.1)
        return {"local_loss": 1.0}

    def evaluate(model, config):
        del config
        return 0.5, {"accuracy": 0.75}

    return train, evaluate


def test_adapter_exchanges_only_head_and_computes_payload_size():
    adapter = PPEHeadAdapter(_FakeDetector())
    assert adapter.trainable_parameter_count == 10
    assert adapter.bytes_per_update == 40
    arrays = adapter.get_parameters()
    assert len(arrays) == 2
    assert [value.shape for value in arrays] == [(2, 4), (2,)]


def test_adapter_rebind_requires_the_same_model_schema():
    adapter = PPEHeadAdapter(_FakeDetector())
    replacement = _FakeDetector()
    adapter.rebind(replacement)
    assert adapter.wrapper is replacement
    with pytest.raises(ValueError, match="schema"):
        adapter.rebind(_FakeDetector({0: "no_helmet", 1: "helmet"}))


def test_adapter_rebind_requires_identical_model_schema():
    adapter = PPEHeadAdapter(_FakeDetector())
    replacement = _FakeDetector()
    adapter.rebind(replacement)
    assert adapter.wrapper is replacement
    with pytest.raises(ValueError, match="schema"):
        adapter.rebind(_FakeDetector({0: "no_helmet", 1: "helmet"}))


def test_flower_client_freezes_backbone_and_returns_schema_metadata():
    model = _FakeDetector()
    adapter = PPEHeadAdapter(model)
    train, evaluate = _callbacks()
    client = PPEHeadFlowerClient(
        adapter,
        site_id="factory-a",
        num_examples=12,
        train_local=train,
        evaluate_local=evaluate,
    )
    before_backbone = [p.detach().clone() for p in adapter.frozen_parameters]
    initial_head = [array.copy() for array in client.get_parameters({})]
    updated, count, metrics = client.fit(initial_head, {})
    assert count == 12
    assert metrics["site_id"] == "factory-a"
    assert metrics["schema_sha256"] == adapter.schema_sha256
    assert any(not np.array_equal(old, new) for old, new in zip(initial_head, updated, strict=True))
    assert all(torch.equal(old, new) for old, new in zip(before_backbone, adapter.frozen_parameters, strict=True))
    loss, eval_count, eval_metrics = client.evaluate(updated, {})
    assert loss == 0.5 and eval_count == 12
    assert eval_metrics["accuracy"] == 0.75


def test_adapter_rejects_wrong_shapes_nonfinite_values_and_non_ppe_models():
    adapter = PPEHeadAdapter(_FakeDetector())
    with pytest.raises(ValueError, match="Expected"):
        adapter.set_parameters([])
    malformed = adapter.get_parameters()
    malformed[0] = np.full((2, 4), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid numeric"):
        adapter.set_parameters(malformed)
    malformed = adapter.get_parameters()
    malformed[0] = malformed[0].astype(np.float64)
    with pytest.raises(ValueError, match="Dtype mismatch"):
        adapter.set_parameters(malformed)
    malformed = adapter.get_parameters()
    malformed[0] = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        adapter.set_parameters(malformed)
    with pytest.raises(RuntimeError, match="frozen PPE backbone"):
        PPEHeadAdapter(_FakeDetector()).assert_backbone_unchanged([])
    with pytest.raises(ValueError):
        PPEHeadAdapter(_FakeDetector({0: "person", 1: "car"}))


def test_flower_client_rejects_callback_that_changes_backbone():
    model = _FakeDetector()
    adapter = PPEHeadAdapter(model)

    def bad_train(core, config):
        del config
        with torch.no_grad():
            core.model[0].weight.add_(1)
        return {}

    client = PPEHeadFlowerClient(
        adapter,
        site_id="factory-a",
        num_examples=1,
        train_local=bad_train,
        evaluate_local=lambda core, config: (0.0, {}),
    )
    with pytest.raises(RuntimeError, match="frozen PPE backbone"):
        client.fit(adapter.get_parameters(), {})


def test_client_evaluation_is_explicitly_disabled_when_no_evaluator_is_given():
    model = _FakeDetector()
    adapter = PPEHeadAdapter(model)
    train, _ = _callbacks()
    client = PPEHeadFlowerClient(
        adapter,
        site_id="factory-a",
        num_examples=1,
        train_local=train,
        evaluate_local=None,
    )
    with pytest.raises(RuntimeError, match="held-out evaluator"):
        client.evaluate(adapter.get_parameters(), {})


def test_fedavg_rejects_missing_or_mixed_client_schemas_before_aggregation():
    strategy = PPEHeadFedAvg(expected_schema_sha256="a" * 64)
    missing = [(object(), SimpleNamespace(metrics={}))]
    mixed = [
        (object(), SimpleNamespace(metrics={"schema_sha256": "a" * 64})),
        (object(), SimpleNamespace(metrics={"schema_sha256": "b" * 64})),
    ]
    with pytest.raises(RuntimeError, match="schema"):
        strategy.aggregate_fit(1, missing, [])
    with pytest.raises(RuntimeError, match="schema"):
        strategy.aggregate_fit(1, mixed, [])
    with pytest.raises(ValueError, match="SHA-256"):
        PPEHeadFedAvg(expected_schema_sha256="not-a-digest")
