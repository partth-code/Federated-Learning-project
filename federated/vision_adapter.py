"""Flower client boundary for frozen-backbone PPE head updates.

The separate bounded demo runner supplies one local YOLO training epoch and
one FedAvg round; this module itself only defines safe parameter exchange.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from flwr.server.strategy import FedAvg

from detector.contracts import validate_ppe_labels


TrainCallback = Callable[[torch.nn.Module, dict[str, Scalar]], Mapping[str, Scalar] | None]
EvaluateCallback = Callable[[torch.nn.Module, dict[str, Scalar]], tuple[float, Mapping[str, Scalar]]]


class PPEHeadAdapter:
    """Expose only the final Ultralytics detection head as Flower ndarrays.

    All participating clients must load checkpoints with the same complete
    parameter schema and ordered class labels. Buffers and the backbone remain
    local and are not sent by this adapter.
    """

    def __init__(self, model: Any) -> None:
        self.wrapper = model
        self.model = getattr(model, "model", model)
        layers = getattr(self.model, "model", None)
        if not isinstance(self.model, torch.nn.Module) or layers is None or len(layers) == 0:
            raise ValueError("Expected a YOLO model with a non-empty model layer sequence")
        self.head = layers[-1]
        head_parameters = list(self.head.named_parameters())
        model_has_trainable_parameters = any(parameter.requires_grad for parameter in self.model.parameters())
        # Inference-loaded Ultralytics checkpoints mark every parameter frozen.
        # In that case expose the detection head for an explicitly head-only
        # client; keep DFL's fixed projection local and frozen.
        self._parameters = [
            (name, parameter)
            for name, parameter in head_parameters
            if (parameter.requires_grad or not model_has_trainable_parameters)
            and name != "dfl.conv.weight"
        ]
        self._trainable_head_ids = {id(parameter) for _, parameter in self._parameters}
        if not self._parameters:
            raise ValueError("Detection head has no trainable parameters")

        raw_names = getattr(model, "names", getattr(self.model, "names", None))
        if isinstance(raw_names, Mapping):
            self.class_names = [str(raw_names[key]) for key in sorted(raw_names, key=lambda key: int(key))]
        elif isinstance(raw_names, (list, tuple)):
            self.class_names = [str(name) for name in raw_names]
        else:
            raise ValueError("PPE model must expose its ordered class names")
        validate_ppe_labels({name.lower() for name in self.class_names})

        full_schema = [
            ("parameter", name, tuple(parameter.shape), str(parameter.dtype))
            for name, parameter in self.model.named_parameters()
        ] + [
            ("buffer", name, tuple(buffer.shape), str(buffer.dtype))
            for name, buffer in self.model.named_buffers()
        ]
        shared_schema = [
            (name, tuple(parameter.shape), str(parameter.dtype))
            for name, parameter in self._parameters
        ]
        schema = {
            "model_type": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
            "head_type": f"{type(self.head).__module__}.{type(self.head).__qualname__}",
            "class_names": self.class_names,
            "full_parameter_schema": full_schema,
            "shared_parameter_schema": shared_schema,
        }
        self.schema_sha256 = hashlib.sha256(
            json.dumps(schema, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for _, parameter in self._parameters)

    @property
    def bytes_per_update(self) -> int:
        return sum(parameter.numel() * parameter.element_size() for _, parameter in self._parameters)

    @property
    def frozen_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if id(parameter) not in self._trainable_head_ids]

    def freeze_backbone(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(id(parameter) in self._trainable_head_ids)

    def get_parameters(self) -> NDArrays:
        return [parameter.detach().cpu().numpy().copy() for _, parameter in self._parameters]

    def rebind(self, model: Any) -> None:
        """Point the adapter at a trainer-replaced model after schema validation."""
        replacement = PPEHeadAdapter(model)
        if replacement.schema_sha256 != self.schema_sha256:
            raise ValueError("Replacement PPE model schema does not match the federated global model")
        self.wrapper = replacement.wrapper
        self.model = replacement.model
        self.head = replacement.head
        self._parameters = replacement._parameters
        self._trainable_head_ids = replacement._trainable_head_ids

    def set_parameters(self, arrays: NDArrays) -> None:
        if len(arrays) != len(self._parameters):
            raise ValueError(f"Expected {len(self._parameters)} head arrays, received {len(arrays)}")
        for (name, parameter), array in zip(self._parameters, arrays, strict=True):
            value = np.asarray(array)
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"Shape mismatch for head parameter {name}")
            if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
                raise ValueError(f"Invalid numeric values for head parameter {name}")
            expected_dtype = parameter.detach().cpu().numpy().dtype
            if value.dtype != expected_dtype:
                raise ValueError(f"Dtype mismatch for head parameter {name}")
            with torch.no_grad():
                parameter.copy_(torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype))

    def backbone_snapshot(self) -> list[torch.Tensor]:
        return [parameter.detach().cpu().clone() for parameter in self.frozen_parameters]

    def assert_backbone_unchanged(self, before: list[torch.Tensor]) -> None:
        after = self.frozen_parameters
        if len(before) != len(after) or any(
            not torch.equal(old, new.detach().cpu()) for old, new in zip(before, after, strict=True)
        ):
            raise RuntimeError("Local callback changed frozen PPE backbone/local parameters; update rejected")


class PPEHeadFlowerClient(NumPyClient):
    """Flower NumPyClient wrapper around caller-supplied local train/eval code."""

    def __init__(
        self,
        adapter: PPEHeadAdapter,
        *,
        site_id: str,
        num_examples: int,
        train_local: TrainCallback,
        evaluate_local: EvaluateCallback | None,
    ) -> None:
        if not site_id.strip():
            raise ValueError("site_id must not be empty")
        if num_examples < 1:
            raise ValueError("num_examples must be positive")
        self.adapter = adapter
        self.site_id = site_id
        self.num_examples = int(num_examples)
        self.train_local = train_local
        self.evaluate_local = evaluate_local

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        del config
        return self.adapter.get_parameters()

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        self.adapter.set_parameters(parameters)
        self.adapter.freeze_backbone()
        before = self.adapter.backbone_snapshot()
        metrics = dict(self.train_local(self.adapter.model, config) or {})
        self.adapter.assert_backbone_unchanged(before)
        metrics.update({"site_id": self.site_id, "schema_sha256": self.adapter.schema_sha256})
        return self.adapter.get_parameters(), self.num_examples, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        self.adapter.set_parameters(parameters)
        if self.evaluate_local is None:
            raise RuntimeError("Client-side PPE evaluation is disabled; use the explicit held-out evaluator")
        before = self.adapter.backbone_snapshot()
        loss, metrics = self.evaluate_local(self.adapter.model, config)
        self.adapter.assert_backbone_unchanged(before)
        if not np.isfinite(loss):
            raise ValueError("Local evaluation loss must be finite")
        result = dict(metrics)
        result.update({"site_id": self.site_id, "schema_sha256": self.adapter.schema_sha256})
        return float(loss), self.num_examples, result


class PPEHeadFedAvg(FedAvg):
    """FedAvg guard that rejects mixed or unexpected PPE model schemas."""

    def __init__(self, *, expected_schema_sha256: str, **kwargs: Any) -> None:
        try:
            valid_digest = len(expected_schema_sha256) == 64 and len(bytes.fromhex(expected_schema_sha256)) == 32
        except ValueError:
            valid_digest = False
        if not valid_digest:
            raise ValueError("expected_schema_sha256 must be a SHA-256 hex digest")
        self.expected_schema_sha256 = expected_schema_sha256
        self.last_aggregated_parameters = None
        self.last_aggregated_fit_metrics = None
        super().__init__(**kwargs)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            raise RuntimeError(f"Round {server_round} has no successful PPE client updates")
        observed = {fit_res.metrics.get("schema_sha256") for _, fit_res in results}
        if observed != {self.expected_schema_sha256}:
            raise RuntimeError(
                f"Round {server_round} contains missing, mixed, or unexpected PPE model schemas"
            )
        aggregated = super().aggregate_fit(server_round, results, failures)
        self.last_aggregated_parameters = aggregated[0]
        self.last_aggregated_fit_metrics = aggregated[1]
        return aggregated
