"""Flower ClientApp with Flower's official SecAgg+ client modifier."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from safesight_secagg.task import data_for_client, train


SITE_IDS = ("factory-a", "factory-b", "factory-c")


def _read_arrays(path: Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return [archive[key].copy() for key in sorted(archive.files)]


class PPEPretrainedUpdateClient(NumPyClient):
    """SecAgg+ client that submits a locally trained PPE head from a temp bundle."""

    def __init__(self, partition_id: int) -> None:
        if not 0 <= partition_id < len(SITE_IDS):
            raise ValueError("Unexpected PPE SecAgg+ simulated-client partition ID")
        self.site_id = SITE_IDS[partition_id]
        update_root = Path(os.environ["SAFESIGHT_PPE_UPDATE_DIR"])
        self.initial = _read_arrays(Path(os.environ["SAFESIGHT_PPE_INITIAL_FILE"]))
        self.local_update = _read_arrays(update_root / f"{self.site_id}.npz")
        counts = json.loads(Path(os.environ["SAFESIGHT_PPE_COUNTS_FILE"]).read_text(encoding="utf-8"))
        self.num_examples = int(counts[self.site_id])
        if self.num_examples < 1 or len(self.initial) != len(self.local_update):
            raise ValueError("PPE SecAgg+ client received invalid local update metadata")
        for initial, local in zip(self.initial, self.local_update, strict=True):
            if initial.shape != local.shape or initial.dtype != local.dtype or not np.isfinite(local).all():
                raise ValueError("PPE SecAgg+ client update failed parameter-schema validation")

    def get_parameters(self, config):
        del config
        return [value.copy() for value in self.initial]

    def fit(self, parameters, config):
        del config
        if len(parameters) != len(self.initial) or any(
            current.shape != expected.shape or current.dtype != expected.dtype
            for current, expected in zip(parameters, self.initial, strict=True)
        ):
            raise ValueError("PPE SecAgg+ server sent an incompatible detection-head schema")
        return [value.copy() for value in self.local_update], self.num_examples, {}


class SecAggDemoClient(NumPyClient):
    def __init__(self, partition_id: int, context: Context) -> None:
        self.partition_id = partition_id
        self.features, self.labels = data_for_client(partition_id)
        self.epochs = int(context.run_config["local-epochs"])
        self.learning_rate = float(context.run_config["learning-rate"])

    def get_parameters(self, config):
        del config
        return [np.zeros((2, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)]

    def fit(self, parameters, config):
        del config
        updated, local_loss = train(parameters, self.features, self.labels, self.epochs, self.learning_rate)
        return updated, len(self.labels), {"local_loss": local_loss}


def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    if os.environ.get("SAFESIGHT_PPE_SECAGG") == "1":
        return PPEPretrainedUpdateClient(partition_id).to_client()
    return SecAggDemoClient(partition_id, context).to_client()


app = ClientApp(client_fn=client_fn, mods=[secaggplus_mod])
