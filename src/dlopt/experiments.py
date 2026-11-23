
"""Run reproducible comparisons between optimization algorithms."""

from pathlib import Path

import pandas as pd
import torch

from .data import load_dataset
from .models import MLP
from .optimizers import get_optimizer
from .training import train_model, evaluate_model
from .plotting import (
    plot_training_curves,
    plot_accuracy_comparison,
)


def run_experiment(config):
    """Run optimizer experiments and save results."""

    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    output_dir = Path(config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = load_dataset(
        n_samples=config.get("n_samples", 1200),
        noise=config.get("noise", 0.20),
        test_size=config.get("test_fraction", 0.25),
        random_state=seed,
    )

    epochs = config.get("epochs", 150)
    hidden_dim = config.get("hidden_dim", 32)
    learning_rates = config.get("learning_rates", {})

    optimizer_names = ["sgd", "momentum", "adam", "lbfgs"]

    results = []
    histories = {}

    for name in optimizer_names:
        torch.manual_seed(seed)

        model = MLP(hidden_dim=hidden_dim)

        optimizer = get_optimizer(
            name,
            model.parameters(),
            learning_rate=learning_rates.get(name, 0.01),
        )

        history = train_model(
            model,
            optimizer,
            x_train,
            y_train,
            epochs=epochs,
        )

        histories[name] = [item["loss"] for item in history]

        metrics = evaluate_model(
            model,
            x_test,
            y_test,
        )

        results.append({
            "optimizer": name,
            "epochs": epochs,
            "final_train_loss": history[-1]["loss"],
            **metrics,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)

    plot_training_curves(
        histories,
        output_dir / "training_curves.png",
    )

    plot_accuracy_comparison(
        results,
        output_dir / "accuracy_comparison.png",
    )

    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_dir.resolve()}")

    return results
