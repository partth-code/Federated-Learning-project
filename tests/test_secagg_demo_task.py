from __future__ import annotations

import numpy as np

from federated.secagg_demo.safesight_secagg.task import data_for_client, evaluate, initial_parameters, train


def test_secagg_demo_workload_is_reproducible_and_trainable() -> None:
    x0, y0 = data_for_client(0)
    x1, y1 = data_for_client(1)
    assert np.array_equal(x0, data_for_client(0)[0])
    assert not np.array_equal(x0, x1)
    initial = initial_parameters()
    initial_loss, _ = evaluate(initial, x0, y0)
    updated, _ = train(initial, x0, y0, epochs=2, learning_rate=0.15)
    updated_loss, accuracy = evaluate(updated, x0, y0)
    assert updated_loss < initial_loss
    assert 0.0 <= accuracy <= 1.0
