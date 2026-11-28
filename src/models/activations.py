"""activations.py - Activation functions with forward and backward passes."""

import numpy as np


class Activation:
    """Base class for activation layers."""

    name = "activation"

    def __init__(self):
        self._cache = None

    def forward(self, z):
        raise NotImplementedError

    def backward(self, d_out):
        raise NotImplementedError


class ReLU(Activation):
    """Rectified Linear Unit: max(0, z)."""

    name = "relu"

    def forward(self, z):
        z_arr = np.asarray(z, dtype=np.float64)
        self._cache = z_arr
        return np.maximum(0.0, z_arr)

    def backward(self, d_out):
        if self._cache is None:
            raise RuntimeError("ReLU.backward called before forward.")
        return d_out * (self._cache > 0.0)


class LeakyReLU(Activation):
    """Leaky ReLU: z if z > 0 else alpha * z."""

    name = "leaky_relu"

    def __init__(self, alpha=0.01):
        super().__init__()
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = float(alpha)

    def forward(self, z):
        z_arr = np.asarray(z, dtype=np.float64)
        self._cache = z_arr
        return np.where(z_arr > 0.0, z_arr, self.alpha * z_arr)

    def backward(self, d_out):
        if self._cache is None:
            raise RuntimeError("LeakyReLU.backward called before forward.")
        grad = np.where(self._cache > 0.0, 1.0, self.alpha)
        return d_out * grad


class Sigmoid(Activation):
    """Numerically stable sigmoid activation."""

    name = "sigmoid"

    def forward(self, z):
        z_arr = np.asarray(z, dtype=np.float64)
        # Split positive and negative values so exp() never overflows
        out = np.empty_like(z_arr, dtype=np.float64)
        pos = z_arr >= 0.0
        out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
        exp_neg = np.exp(z_arr[~pos])
        out[~pos] = exp_neg / (1.0 + exp_neg)
        self._cache = out
        return out

    def backward(self, d_out):
        if self._cache is None:
            raise RuntimeError("Sigmoid.backward called before forward.")
        s = self._cache
        return d_out * s * (1.0 - s)


class Tanh(Activation):
    """Hyperbolic tangent activation."""

    name = "tanh"

    def forward(self, z):
        z_arr = np.asarray(z, dtype=np.float64)
        out = np.tanh(z_arr)
        self._cache = out
        return out

    def backward(self, d_out):
        if self._cache is None:
            raise RuntimeError("Tanh.backward called before forward.")
        t = self._cache
        return d_out * (1.0 - t * t)


class Softmax(Activation):
    """Row-wise softmax using max subtraction for stability."""

    name = "softmax"

    def forward(self, z):
        z_arr = np.asarray(z, dtype=np.float64)
        shifted = z_arr - np.max(z_arr, axis=-1, keepdims=True)
        exp_z = np.exp(shifted)
        probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        self._cache = probs
        return probs

    def backward(self, d_out):
        if self._cache is None:
            raise RuntimeError("Softmax.backward called before forward.")
        s = self._cache
        dot = np.sum(d_out * s, axis=-1, keepdims=True)
        return s * (d_out - dot)


_ACTIVATIONS = {
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
}


def get_activation(name):
    """Create an activation instance from a string name."""
    if isinstance(name, Activation):
        return name
    key = name.strip().lower()
    if key not in _ACTIVATIONS:
        supported = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(f"Unsupported activation '{name}'. Choose from: {supported}")
    return _ACTIVATIONS[key]()
