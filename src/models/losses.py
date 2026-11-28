"""losses.py - Softmax cross-entropy loss and classification accuracy."""

import numpy as np


def softmax(logits):
    """Numerically stable softmax along the last axis."""
    logits_arr = np.asarray(logits, dtype=np.float64)
    shifted = logits_arr - np.max(logits_arr, axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def _normalize_targets(targets, num_classes, batch_size):
    """Convert 1D class indices or 2D one-hot vectors into 1D integer indices."""
    t_arr = np.asarray(targets)
    if t_arr.ndim == 2:
        if t_arr.shape != (batch_size, num_classes):
            raise ValueError(
                f"Expected one-hot targets of shape {(batch_size, num_classes)}, got {t_arr.shape}."
            )
        return np.argmax(t_arr, axis=1).astype(np.int64)
    if t_arr.ndim == 1:
        if t_arr.shape[0] != batch_size:
            raise ValueError(f"Expected {batch_size} target labels, got {t_arr.shape[0]}.")
        idx = t_arr.astype(np.int64)
        if np.any(idx < 0) or np.any(idx >= num_classes):
            raise ValueError(f"Target class indices must lie in [0, {num_classes - 1}].")
        return idx
    raise ValueError(f"Targets must be 1D indices or 2D one-hot, got {t_arr.shape}.")


class SoftmaxCrossEntropyLoss:
    """Combined log-softmax and cross-entropy loss."""

    def __init__(self):
        self._probs = None
        self._target_idx = None

    def forward(self, logits, targets):
        """Compute mean cross-entropy loss using the log-sum-exp trick."""
        logits_arr = np.asarray(logits, dtype=np.float64)
        if logits_arr.ndim != 2:
            raise ValueError(f"Expected 2D logits (B, C), got {logits_arr.shape}.")

        batch_size, num_classes = logits_arr.shape
        target_idx = _normalize_targets(targets, num_classes, batch_size)

        shifted = logits_arr - np.max(logits_arr, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp

        self._probs = np.exp(log_probs)
        self._target_idx = target_idx

        nll = -log_probs[np.arange(batch_size), target_idx]
        return float(np.mean(nll))

    def backward(self):
        """Gradient of loss with respect to logits: (probs - one_hot) / batch_size."""
        if self._probs is None or self._target_idx is None:
            raise RuntimeError("SoftmaxCrossEntropyLoss.backward called before forward.")
        batch_size = self._probs.shape[0]
        d_logits = self._probs.copy()
        d_logits[np.arange(batch_size), self._target_idx] -= 1.0
        d_logits /= batch_size
        return d_logits


def classification_accuracy(logits_or_probs, targets):
    """Return top-1 classification accuracy in [0.0, 1.0]."""
    arr = np.asarray(logits_or_probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D logits/probs (B, C), got {arr.shape}.")
    target_idx = _normalize_targets(targets, arr.shape[1], arr.shape[0])
    preds = np.argmax(arr, axis=1)
    return float(np.mean(preds == target_idx))
