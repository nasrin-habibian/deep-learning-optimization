"""Training and evaluation routines for optimizer experiments."""

import torch
import torch.nn as nn


def train_model(
    model,
    optimizer,
    X_train,
    y_train,
    epochs=100,
):
    """Train a model and return its training history."""

    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        history.append({
            "epoch": epoch + 1,
            "loss": float(loss.item()),
        })

    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate classification accuracy and loss."""

    criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        loss = criterion(outputs, y_test)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == y_test).float().mean()

    return {
        "test_loss": float(loss.item()),
        "test_accuracy": float(accuracy.item()),
    }
