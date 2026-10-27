"""Deterministic NumPy classifier/data for the SecAgg+ systems demo."""

from __future__ import annotations

import numpy as np


CLIENTS = ("factory-a", "factory-b", "factory-c")


def data_for_client(client_index: int, count: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(700 + client_index)
    shift = np.asarray([client_index * 0.65, -client_index * 0.3], dtype=np.float32)
    features = rng.normal(size=(count, 2)).astype(np.float32) + shift
    labels = ((features[:, 0] + 0.7 * features[:, 1]) > 0).astype(np.int64)
    return features, labels


def held_out_data(count: int = 240) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9700)
    features = rng.normal(size=(count, 2)).astype(np.float32)
    labels = ((features[:, 0] + 0.7 * features[:, 1]) > 0).astype(np.int64)
    return features, labels


def initial_parameters() -> list[np.ndarray]:
    rng = np.random.default_rng(13)
    return [rng.normal(0, 0.1, size=(2, 2)).astype(np.float32), np.zeros(2, dtype=np.float32)]


def train(parameters: list[np.ndarray], features: np.ndarray, labels: np.ndarray, epochs: int, learning_rate: float) -> tuple[list[np.ndarray], float]:
    weights, bias = (np.array(value, dtype=np.float32, copy=True) for value in parameters)
    for _ in range(epochs):
        logits = features @ weights + bias
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(labels)), labels] -= 1.0
        probs /= len(labels)
        weights -= learning_rate * (features.T @ probs)
        bias -= learning_rate * probs.sum(axis=0)
    return [weights, bias], loss(parameters=[weights, bias], features=features, labels=labels)


def loss(parameters: list[np.ndarray], features: np.ndarray, labels: np.ndarray) -> float:
    weights, bias = parameters
    logits = features @ weights + bias
    logits -= logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(logits).sum(axis=1))
    return float(np.mean(logsumexp - logits[np.arange(len(labels)), labels]))


def evaluate(parameters: list[np.ndarray], features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    weights, bias = parameters
    logits = features @ weights + bias
    predictions = logits.argmax(axis=1)
    return loss(parameters, features, labels), float(np.mean(predictions == labels))
