
"""Optimizer configurations for controlled training comparisons."""

import torch


def get_optimizer(
    name,
    parameters,
    learning_rate=0.01,
    weight_decay=0.0,
):
    """Create an optimizer by name."""

    name = name.lower()

    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if name == "momentum":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if name == "lbfgs":
        return torch.optim.LBFGS(
            parameters,
            lr=learning_rate,
            max_iter=10,
            history_size=10,
        )

    raise ValueError(
        f"Unknown optimizer: {name}. "
        "Choose sgd, momentum, adam, or lbfgs."
    )
