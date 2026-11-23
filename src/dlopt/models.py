"""Neural network models used in optimizer experiments."""

import torch.nn as nn


class MLP(nn.Module):
    """A small multilayer perceptron for binary classification."""

    def __init__(self, input_dim=2, hidden_dim=32, output_dim=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)
