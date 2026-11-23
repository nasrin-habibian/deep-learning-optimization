"""Run reproducible comparisons between optimization algorithms."""

import torch

from .data import load_dataset
from .models import MLP
from .optimizers import get_optimizer
from .training import train_model, evaluate_model


def run_comparison(
    optimizer_names=None,
    epochs=100,
    learning_rate=0.01,
    seed=42,
):
    """Train and evaluate one model for each optimizer."""

    if optimizer_names is None:
        optimizer_names = ["sgd", "momentum", "adam", "lbfgs"]

    X_train, X_test, y_train, y_test = load_dataset()

    results = []

    for name in optimizer_names:
        torch.manual_seed(seed)

        model = MLP()
        optimizer = get_optimizer(
            name,
            model.parameters(),
            learning_rate=learning_rate,
        )

        history = train_model(
            model,
            optimizer,
            X_train,
            y_train,
            epochs=epochs,
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        results.append({
            "optimizer": name,
            "epochs": epochs,
            "final_train_loss": history[-1]["loss"],
            **metrics,
        })

    return results
