"""Graph Neural Network for power grid state estimation and price forecasting."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """D^{-1/2} A D^{-1/2} normalization."""
    A_hat = A + np.eye(A.shape[0])
    D = np.diag(A_hat.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-8))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


class GraphConvLayer:
    """GCN layer: H = sigma(A_hat * H * W)."""

    def __init__(self, in_features: int, out_features: int,
                 activation: str = "relu"):
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        self._act = _relu if activation == "relu" else (lambda x: np.tanh(x))

    def forward(self, X: np.ndarray, A_hat: np.ndarray) -> np.ndarray:
        """X: (n_nodes, in_features), A_hat: normalized adjacency."""
        return self._act(A_hat @ X @ self.W + self.b)

    def backward(self, X: np.ndarray, A_hat: np.ndarray,
                 grad_out: np.ndarray, lr: float = 0.01) -> np.ndarray:
        """Simplified gradient update."""
        AX = A_hat @ X
        self.W -= lr * AX.T @ grad_out
        self.b -= lr * grad_out.mean(axis=0)
        return grad_out @ self.W.T  # pass gradient to previous layer


class PowerGridGNN:
    """Multi-layer GCN for power grid node-level predictions."""

    def __init__(self, n_node_features: int,
                 hidden_dims: List[int] = None,
                 output_dim: int = 1):
        dims = [n_node_features] + (hidden_dims or [64, 32]) + [output_dim]
        self._layers = []
        for i in range(len(dims) - 2):
            self._layers.append(GraphConvLayer(dims[i], dims[i + 1], "relu"))
        # Output layer (linear)
        self._out_W = np.random.randn(dims[-2], output_dim) * 0.01
        self._out_b = np.zeros(output_dim)

    def forward(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Forward pass. Returns predictions of shape (n_nodes, output_dim)."""
        A_hat = _normalize_adjacency(A)
        H = X.copy()
        for layer in self._layers:
            H = layer.forward(H, A_hat)
        return H @ self._out_W + self._out_b

    def fit(self, node_features: np.ndarray, adjacency: np.ndarray,
            targets: np.ndarray, epochs: int = 100, lr: float = 0.01,
            seed: int = 42) -> List[float]:
        """Train via MSE loss with gradient descent."""
        np.random.seed(seed)
        A_hat = _normalize_adjacency(adjacency)
        losses = []
        for epoch in range(epochs):
            # Forward
            H = node_features.copy()
            activations = [H]
            for layer in self._layers:
                H = layer.forward(H, A_hat)
                activations.append(H)
            preds = H @ self._out_W + self._out_b
            loss = float(np.mean((preds - targets) ** 2))
            losses.append(loss)

            # Backward (simplified: only update output layer)
            grad = 2 * (preds - targets) / len(targets)
            self._out_W -= lr * H.T @ grad
            self._out_b -= lr * grad.mean(axis=0)

            # Gradient to last hidden layer
            grad_H = grad @ self._out_W.T
            for layer in reversed(self._layers):
                grad_H = layer.backward(activations[self._layers.index(layer)],
                                        A_hat, grad_H, lr)
        return losses

    def predict(self, node_features: np.ndarray,
                adjacency: np.ndarray) -> np.ndarray:
        return self.forward(node_features, adjacency)

    def node_embeddings(self, node_features: np.ndarray,
                         adjacency: np.ndarray) -> np.ndarray:
        """Return penultimate layer embeddings."""
        A_hat = _normalize_adjacency(adjacency)
        H = node_features.copy()
        for layer in self._layers[:-1]:
            H = layer.forward(H, A_hat)
        # Last layer
        if self._layers:
            H = self._layers[-1].forward(H, A_hat)
        return H


class GridFeatureExtractor:
    """Extract node and edge features from power grid time series."""

    def extract_node_features(self, buses: List[dict],
                               time_series: Optional[dict] = None) -> np.ndarray:
        """Build feature matrix (n_buses, n_features) from bus attributes."""
        features = []
        for bus in buses:
            f = [
                float(bus.get("P_gen", 0.0)),
                float(bus.get("P_load", 0.0)),
                float(bus.get("voltage_pu", 1.0)),
                float(bus.get("type_pq", bus.get("bus_type", "PQ") == "PQ")),
                float(bus.get("type_pv", bus.get("bus_type", "PV") == "PV")),
            ]
            if time_series and str(bus.get("bus_id")) in time_series:
                ts = time_series[str(bus["bus_id"])]
                f += [float(np.mean(ts)), float(np.std(ts)),
                      float(np.max(ts)), float(np.min(ts))]
            features.append(f)
        return np.array(features)

    def build_adjacency(self, branches: List[dict]) -> np.ndarray:
        """Build adjacency matrix from branch list."""
        all_buses = set()
        for br in branches:
            all_buses.add(br["from_bus"])
            all_buses.add(br["to_bus"])
        buses = sorted(all_buses)
        idx = {b: i for i, b in enumerate(buses)}
        n = len(buses)
        A = np.zeros((n, n))
        for br in branches:
            i, j = idx[br["from_bus"]], idx[br["to_bus"]]
            w = br.get("susceptance", 1.0)
            A[i, j] = w
            A[j, i] = w
        return A
