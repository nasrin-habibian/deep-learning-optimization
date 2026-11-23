
"""Training and evaluation routines for optimizer experiments."""

import torch
import torch.nn as nn


def train_model(
    model,
    optimizer,
    x_train,
    y_train,
    epochs=100,
):
    """Train a model and return its training history."""

    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(epochs):
        model.train()

        if isinstance(optimizer, torch.optim.LBFGS):

            def closure():
                optimizer.zero_grad()
                outputs = model(x_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                return loss

            optimizer.step(closure)

        else:
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        # Record the loss after the optimizer update.
        model.eval()
        with torch.no_grad():
            outputs = model(x_train)
            train_loss = criterion(outputs, y_train)

        history.append({
            "epoch": epoch + 1,
            "loss": float(train_loss.item()),
        })

    return history


def evaluate_model(model, x_test, y_test):
    """Evaluate classification accuracy and loss."""

    criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        loss = criterion(outputs, y_test)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == y_test).float().mean()

    return {
        "test_loss": float(loss.item()),
        "test_accuracy": float(accuracy.item()),
    }
