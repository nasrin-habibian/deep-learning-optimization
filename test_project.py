import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dlopt.data import make_two_moons, train_test_split
from dlopt.models import MLP
from dlopt.optimizers import build_optimizer
from dlopt.training import train_model


class ProjectTests(unittest.TestCase):
    def test_dataset_shapes_and_labels(self):
        x, y = make_two_moons(n_samples=100, seed=1)
        self.assertEqual(tuple(x.shape), (100, 2))
        self.assertEqual(tuple(y.shape), (100,))
        self.assertEqual(set(y.tolist()), {0, 1})

    def test_split_sizes(self):
        x, y = make_two_moons(n_samples=100, seed=1)
        x_train, y_train, x_test, y_test = train_test_split(x, y, 0.2, 2)
        self.assertEqual(len(x_train) + len(x_test), 100)
        self.assertEqual(len(y_train), len(x_train))
        self.assertEqual(len(y_test), len(x_test))

    def test_all_optimizers_train(self):
        x, y = make_two_moons(n_samples=120, seed=3)
        for name, lr in [("sgd", 0.05), ("momentum", 0.05), ("adam", 0.01), ("lbfgs", 0.5)]:
            torch.manual_seed(7)
            model = MLP(hidden_dim=12)
            optimizer = build_optimizer(name, model, lr)
            history = train_model(model, optimizer, x, y, epochs=3)
            self.assertEqual(len(history), 3)
            self.assertTrue(torch.isfinite(torch.tensor(history[-1]["train_loss"])))


if __name__ == "__main__":
    unittest.main()
