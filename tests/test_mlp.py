"""Unit tests for the MLP model and TSP route decoder."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.models import (
    LeakyReLU,
    MLP,
    ReLU,
    Sigmoid,
    Softmax,
    SoftmaxCrossEntropyLoss,
    Tanh,
    classification_accuracy,
)
from src.routing import decode_tour, evaluate_routing, tour_length, validate_tour


def _rel_error(analytic, numeric):
    denom = np.maximum(1e-8, np.abs(analytic) + np.abs(numeric))
    return float(np.max(np.abs(analytic - numeric) / denom))


def _numerical_grad_array(fn, arr, eps=1e-6):
    grad = np.zeros_like(arr, dtype=np.float64)
    it = np.nditer(arr, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = float(arr[idx])
        arr[idx] = orig + eps
        pos = fn()
        arr[idx] = orig - eps
        neg = fn()
        arr[idx] = orig
        grad[idx] = (pos - neg) / (2.0 * eps)
        it.iternext()
    return grad


class TestActivations(unittest.TestCase):
    def test_activation_gradients_match_finite_differences(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0.0, 1.0, size=(4, 5))
        x[np.abs(x) < 0.05] = 0.2
        upstream = rng.normal(0.0, 1.0, size=(4, 5))

        for act in [ReLU(), LeakyReLU(0.1), Sigmoid(), Tanh(), Softmax()]:
            out = act.forward(x)
            self.assertEqual(out.shape, x.shape)
            analytic = act.backward(upstream)

            x_work = x.copy()
            numeric = _numerical_grad_array(
                lambda: float(np.sum(act.forward(x_work) * upstream)),
                x_work,
            )
            self.assertLess(
                _rel_error(analytic, numeric),
                1e-6,
                msg=f"Gradient check failed for {act.name}",
            )

    def test_sigmoid_and_softmax_numerical_stability(self):
        extreme = np.array([[-1000.0, 0.0, 1000.0], [950.0, -950.0, 5.0]])
        sig = Sigmoid().forward(extreme)
        sm = Softmax().forward(extreme)
        self.assertTrue(np.all(np.isfinite(sig)))
        self.assertTrue(np.all(np.isfinite(sm)))
        np.testing.assert_allclose(np.sum(sm, axis=1), np.ones(2), atol=1e-12)


class TestLoss(unittest.TestCase):
    def test_loss_stability_and_targets_equivalence(self):
        loss_fn = SoftmaxCrossEntropyLoss()
        logits = np.array([[1000.0, -1000.0, 0.0], [-500.0, 500.0, 499.0]])
        idx_targets = np.array([0, 1])
        onehot_targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        l_idx = loss_fn.forward(logits, idx_targets)
        g_idx = loss_fn.backward()
        l_oh = loss_fn.forward(logits, onehot_targets)
        g_oh = loss_fn.backward()

        self.assertTrue(np.isfinite(l_idx))
        self.assertAlmostEqual(l_idx, l_oh, places=12)
        np.testing.assert_allclose(g_idx, g_oh, atol=1e-12)

    def test_loss_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(11)
        logits = rng.normal(0.0, 1.5, size=(5, 4))
        targets = np.array([0, 3, 1, 2, 0])
        loss_fn = SoftmaxCrossEntropyLoss()

        _ = loss_fn.forward(logits, targets)
        analytic = loss_fn.backward()

        logits_work = logits.copy()
        numeric = _numerical_grad_array(
            lambda: loss_fn.forward(logits_work, targets),
            logits_work,
        )
        self.assertLess(_rel_error(analytic, numeric), 1e-6)


class TestMLP(unittest.TestCase):
    def test_architecture_shapes_and_weights_interface(self):
        model = MLP(input_dim=60, hidden_dims=(128, 64), output_dim=20, activation="relu", seed=42)
        counts = model.num_parameters()
        self.assertEqual(counts["weights"], 60 * 128 + 128 * 64 + 64 * 20)
        self.assertEqual(counts["biases"], 128 + 64 + 20)
        self.assertEqual(counts["total"], 17364)
        self.assertIn("Dense(60 -> 128)", model.summary())

        X = np.random.default_rng(1).normal(size=(8, 60))
        logits = model.forward(X)
        probs = model.predict_proba(X)
        preds = model.predict(X)
        self.assertEqual(logits.shape, (8, 20))
        self.assertEqual(probs.shape, (8, 20))
        self.assertEqual(preds.shape, (8,))
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(8), atol=1e-12)

        model.eval()
        self.assertFalse(model.training)
        model.train()
        self.assertTrue(model.training)

        snapshot = model.get_weights()
        for param, _ in model.parameters():
            param += 1.0
        self.assertFalse(np.allclose(model.layers[0].W, snapshot["layer0.W"]))

        model.set_weights(snapshot)
        np.testing.assert_allclose(model.layers[0].W, snapshot["layer0.W"])

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "weights.npz"
            model.save_weights(ckpt)
            for param, _ in model.parameters():
                param.fill(0.0)
            model.load_weights(ckpt)
            np.testing.assert_allclose(model.layers[0].W, snapshot["layer0.W"])

    def test_full_mlp_backprop_gradient_check(self):
        rng = np.random.default_rng(23)
        X = rng.normal(0.0, 0.8, size=(6, 5))
        y = np.array([0, 2, 1, 2, 0, 1])

        for act in ["sigmoid", "tanh", "relu"]:
            model = MLP(input_dim=5, hidden_dims=(4, 3), output_dim=3, activation=act, seed=123)
            logits = model.forward(X)
            _ = model.compute_loss(logits, y)
            dX_analytic = model.backward()

            for name, param, grad in model.named_parameters():
                numeric = _numerical_grad_array(
                    lambda: model.compute_loss(model.forward(X), y),
                    param,
                )
                self.assertLess(
                    _rel_error(grad, numeric),
                    1e-5,
                    msg=f"Gradient mismatch on {name} for activation={act}",
                )

            X_work = X.copy()
            dX_numeric = _numerical_grad_array(
                lambda: model.compute_loss(model.forward(X_work), y),
                X_work,
            )
            self.assertLess(
                _rel_error(dX_analytic, dX_numeric),
                1e-5,
                msg=f"Input gradient mismatch for activation={act}",
            )

            model.zero_grad()
            for _, _, grad in model.named_parameters():
                self.assertTrue(np.all(grad == 0.0))

    def test_controlled_learning_demonstration(self):
        rng = np.random.default_rng(99)
        X = rng.normal(0.0, 1.0, size=(24, 12))
        y = np.arange(24) % 4

        model = MLP(input_dim=12, hidden_dims=(32, 16), output_dim=4, activation="relu", seed=99)
        initial_loss = model.compute_loss(model.forward(X), y)

        lr = 0.25
        for _ in range(150):
            model.zero_grad()
            logits = model.forward(X)
            _ = model.compute_loss(logits, y)
            model.backward()
            for param, grad in model.parameters():
                param -= lr * grad

        final_logits = model.forward(X)
        final_loss = model.compute_loss(final_logits, y)
        final_acc = classification_accuracy(final_logits, y)

        self.assertLess(final_loss, initial_loss * 0.1)
        self.assertGreaterEqual(final_acc, 0.95)


class TestRoutingDecoder(unittest.TestCase):
    def test_decode_enforces_visited_mask_and_returns_valid_tour(self):
        class AlwaysZeroModel:
            input_dim = 15
            output_dim = 5
            training = True

            def eval(self):
                self.training = False

            def train(self):
                self.training = True

            def forward(self, X):
                return np.tile(np.array([[100.0, 1.0, 2.0, 3.0, 4.0]]), (X.shape[0], 1))

        D = np.array([
            [0.0, 2.0, 5.0, 4.0, 3.0],
            [2.0, 0.0, 1.0, 6.0, 4.0],
            [5.0, 1.0, 0.0, 2.0, 5.0],
            [4.0, 6.0, 2.0, 0.0, 1.0],
            [3.0, 4.0, 5.0, 1.0, 0.0],
        ])
        dummy = AlwaysZeroModel()
        tour = decode_tour(dummy, D, start=0)
        self.assertTrue(dummy.training)
        self.assertEqual(tour, [0, 4, 3, 2, 1])

        valid, msg = validate_tour(D, tour, start=0)
        self.assertTrue(valid, msg=msg)
        self.assertAlmostEqual(tour_length(D, tour), 9.0)

    def test_validate_tour_catches_constraint_violations(self):
        D = np.ones((4, 4)) - np.eye(4)
        self.assertFalse(validate_tour(D, [0, 1, 2])[0])
        self.assertFalse(validate_tour(D, [1, 0, 2, 3])[0])
        self.assertFalse(validate_tour(D, [0, 1, 1, 3])[0])

        D_broken = D.copy()
        D_broken[3, 0] = np.inf
        self.assertFalse(validate_tour(D_broken, [0, 1, 2, 3])[0])

    def test_evaluate_routing_with_mlp(self):
        rng = np.random.default_rng(5)
        n = 6
        mats = []
        for _ in range(5):
            A = rng.uniform(1.0, 20.0, size=(n, n))
            D = 0.5 * (A + A.T)
            np.fill_diagonal(D, 0.0)
            mats.append(D)
        mats_arr = np.array(mats)

        model = MLP(input_dim=3 * n, hidden_dims=(32,), output_dim=n, seed=5)
        ref_tours = np.tile(np.arange(n), (5, 1))
        report = evaluate_routing(model, mats_arr, ref_tours=ref_tours, start=0)
        self.assertEqual(report["num_instances"], 5)
        self.assertEqual(report["valid_tour_rate"], 100.0)
        self.assertGreater(report["mean_model_cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
