import torch


def build_optimizer(name, model, learning_rate):
    """Construct one of the optimizers used in the experiment."""
    key = name.lower()
    if key == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    if key == "momentum":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if key == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if key == "lbfgs":
        return torch.optim.LBFGS(
            model.parameters(),
            lr=learning_rate,
            max_iter=1,
            history_size=20,
            line_search_fn="strong_wolfe",
        )
    raise ValueError(f"Unknown optimizer: {name}")
