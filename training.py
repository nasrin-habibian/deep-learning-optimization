import math
import torch
import torch.nn as nn


def parameter_gradient_norm(model):
    """Return the Euclidean norm of all currently available parameter gradients."""
    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_norm += float(torch.sum(parameter.grad.detach() ** 2).item())
    return math.sqrt(squared_norm)


def evaluate(model, x, y, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y).item()
        accuracy = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, accuracy


def train_model(model, optimizer, x_train, y_train, epochs):
    """Full-batch training with epoch-level diagnostics."""
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        if isinstance(optimizer, torch.optim.LBFGS):
            # L-BFGS requires a closure because it may reevaluate the objective
            # during its line search.
            gradient_norm_holder = {}

            def closure():
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_train)
                loss = criterion(logits, y_train)
                loss.backward()
                gradient_norm_holder["value"] = parameter_gradient_norm(model)
                return loss

            optimizer.step(closure)
            grad_norm = gradient_norm_holder.get("value", float("nan"))
        else:
            logits = model(x_train)
            loss = criterion(logits, y_train)
            loss.backward()
            grad_norm = parameter_gradient_norm(model)
            optimizer.step()

        train_loss, train_accuracy = evaluate(model, x_train, y_train, criterion)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "gradient_norm": grad_norm,
        })

    return history
