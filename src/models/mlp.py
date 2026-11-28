"""mlp.py - Member 2: Multilayer Perceptron (MLP) model implementation."""

from pathlib import Path
import numpy as np

from src.models.activations import get_activation
from src.models.layers import DenseLayer
from src.models.losses import SoftmaxCrossEntropyLoss, softmax


def _resolve_init(init, activation_name, is_output):
    if init.strip().lower() != "auto":
        return init
    if is_output:
        return "xavier"
    if activation_name in ("relu", "leaky_relu"):
        return "he"
    return "xavier"


class MLP:
    """Modular Multilayer Perceptron for TSP next-city classification.

    Structure: Input(input_dim) -> [Dense -> Activation] x L -> Dense(output_dim).
    """

    def __init__(
        self,
        input_dim=60,
        hidden_dims=(128, 64),
        output_dim=20,
        activation="relu",
        seed=42,
        init="auto",
    ):
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive integers.")
        hidden_list = [int(h) for h in hidden_dims]
        if not hidden_list or any(h <= 0 for h in hidden_list):
            raise ValueError("hidden_dims must contain at least one positive layer width.")

        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(hidden_list)
        self.output_dim = int(output_dim)
        self.activation_name = (
            activation.strip().lower() if isinstance(activation, str) else activation.name
        )
        self.seed = seed
        self.init = init
        self.training = True

        rng = np.random.default_rng(seed)
        dims = [self.input_dim, *self.hidden_dims, self.output_dim]

        self.layers = []
        self.activations = []

        num_transitions = len(dims) - 1
        for idx in range(num_transitions):
            is_output = idx == num_transitions - 1
            layer_init = _resolve_init(init, self.activation_name, is_output=is_output)
            self.layers.append(DenseLayer(dims[idx], dims[idx + 1], init=layer_init, rng=rng))
            if not is_output:
                self.activations.append(get_activation(activation))

        self.loss_fn = SoftmaxCrossEntropyLoss()

    def train(self):
        """Set model to training mode."""
        self.training = True
        return self

    def eval(self):
        """Set model to evaluation mode."""
        self.training = False
        return self

    def forward(self, X):
        """Run forward pass and return raw logits of shape (B, output_dim)."""
        out = np.asarray(X, dtype=np.float64)
        if out.ndim != 2 or out.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected 2D input with {self.input_dim} features, got {out.shape}."
            )
        for idx, layer in enumerate(self.layers):
            out = layer.forward(out)
            if idx < len(self.activations):
                out = self.activations[idx].forward(out)
        return out

    def predict_proba(self, X):
        """Return softmax probabilities of shape (B, output_dim)."""
        return softmax(self.forward(X))

    def predict(self, X):
        """Return predicted next-city indices of shape (B,)."""
        return np.argmax(self.forward(X), axis=1)

    def compute_loss(self, logits, targets):
        """Compute mean softmax cross-entropy loss."""
        return self.loss_fn.forward(logits, targets)

    def backward(self, d_logits=None):
        """Backpropagate gradients through all layers and return dL/dX."""
        grad = self.loss_fn.backward() if d_logits is None else np.asarray(d_logits, dtype=np.float64)
        last_idx = len(self.layers) - 1
        grad = self.layers[last_idx].backward(grad)
        for idx in range(last_idx - 1, -1, -1):
            grad = self.activations[idx].backward(grad)
            grad = self.layers[idx].backward(grad)
        return grad

    def zero_grad(self):
        """Clear all layer gradients."""
        for layer in self.layers:
            layer.zero_grad()

    def named_parameters(self):
        """Return [(name, param, grad), ...] for optimizer updates."""
        items = []
        for idx, layer in enumerate(self.layers):
            for pname, param, grad in layer.parameters_and_grads():
                items.append((f"layer{idx}.{pname}", param, grad))
        return items

    def parameters(self):
        """Return [(param, grad), ...] pairs for in-place optimizer steps."""
        return [(param, grad) for _, param, grad in self.named_parameters()]

    def get_weights(self):
        """Return a deep copy of all trainable weights."""
        return {name: param.copy() for name, param, _ in self.named_parameters()}

    def set_weights(self, weights):
        """Restore weights in-place from a dictionary returned by get_weights()."""
        named = self.named_parameters()
        expected = {name for name, _, _ in named}
        if set(weights.keys()) != expected:
            raise ValueError(
                f"Weight keys mismatch. Expected {sorted(expected)}, got {sorted(weights.keys())}."
            )
        for name, param, _ in named:
            arr = np.asarray(weights[name], dtype=np.float64)
            if arr.shape != param.shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': expected {param.shape}, got {arr.shape}."
                )
            param[...] = arr

    def save_weights(self, path):
        """Save weights to an .npz checkpoint file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, **self.get_weights())
        return target

    def load_weights(self, path):
        """Load weights from an .npz checkpoint file."""
        with np.load(path) as data:
            self.set_weights({key: data[key] for key in data.files})

    def num_parameters(self):
        """Return weight, bias, and total parameter counts."""
        weights = sum(layer.W.size for layer in self.layers)
        biases = sum(layer.b.size for layer in self.layers)
        return {"weights": weights, "biases": biases, "total": weights + biases}

    def summary(self):
        """Return a readable architecture summary."""
        lines = [
            f"MLP(input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, "
            f"output_dim={self.output_dim}, activation='{self.activation_name}')"
        ]
        for idx, layer in enumerate(self.layers):
            act = self.activations[idx].name if idx < len(self.activations) else "linear (logits)"
            p_count = layer.W.size + layer.b.size
            lines.append(
                f"  [{idx}] Dense({layer.in_features} -> {layer.out_features}) + {act} "
                f"| params={p_count}"
            )
        counts = self.num_parameters()
        lines.append(
            f"Total parameters: {counts['total']} "
            f"(weights={counts['weights']}, biases={counts['biases']})"
        )
        return "\n".join(lines)
