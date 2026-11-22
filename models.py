import torch.nn as nn


class MLP(nn.Module):
    """A compact multilayer perceptron for two-dimensional classification."""

    def __init__(self, input_dim=2, hidden_dim=64, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)
