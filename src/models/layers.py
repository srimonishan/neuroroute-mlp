"""layers.py - Dense (fully connected) linear layer."""

import numpy as np


def _init_weights(in_features, out_features, init, rng):
    """Initialize weight matrix W of shape (in_features, out_features)."""
    scheme = init.strip().lower()
    if scheme == "he":
        std = np.sqrt(2.0 / in_features)
        return rng.normal(0.0, std, size=(in_features, out_features)).astype(np.float64)
    if scheme == "xavier":
        limit = np.sqrt(6.0 / (in_features + out_features))
        return rng.uniform(-limit, limit, size=(in_features, out_features)).astype(np.float64)
    if scheme == "lecun":
        std = np.sqrt(1.0 / in_features)
        return rng.normal(0.0, std, size=(in_features, out_features)).astype(np.float64)
    if scheme == "zeros":
        return np.zeros((in_features, out_features), dtype=np.float64)
    raise ValueError(f"Unsupported weight init scheme '{init}'.")


class DenseLayer:
    """Fully connected layer computing Z = X @ W + b."""

    def __init__(self, in_features, out_features, init="he", rng=None):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive integers.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.init = init

        generator = rng if rng is not None else np.random.default_rng()
        self.W = _init_weights(self.in_features, self.out_features, init, generator)
        self.b = np.zeros(self.out_features, dtype=np.float64)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._input_cache = None

    def forward(self, X):
        """Forward pass: Z = X @ W + b."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.in_features:
            raise ValueError(
                f"Expected 2D input with {self.in_features} features, got {X_arr.shape}."
            )
        self._input_cache = X_arr
        return X_arr @ self.W + self.b

    def backward(self, dZ):
        """Backward pass: compute dW, db and return dX."""
        if self._input_cache is None:
            raise RuntimeError("DenseLayer.backward called before forward.")
        dZ_arr = np.asarray(dZ, dtype=np.float64)
        expected = (self._input_cache.shape[0], self.out_features)
        if dZ_arr.shape != expected:
            raise ValueError(f"Expected dZ shape {expected}, got {dZ_arr.shape}.")

        # Update in-place so references from model.parameters() stay valid
        self.dW[...] = self._input_cache.T @ dZ_arr
        self.db[...] = np.sum(dZ_arr, axis=0)
        return dZ_arr @ self.W.T

    def zero_grad(self):
        """Reset gradients to zero."""
        self.dW.fill(0.0)
        self.db.fill(0.0)

    def parameters_and_grads(self):
        """Return [(name, param, grad)] for weights and biases."""
        return [("W", self.W, self.dW), ("b", self.b, self.db)]
