import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import make_two_moons, train_test_split
from .models import MLP
from .optimizers import build_optimizer
from .training import train_model, evaluate
from .plotting import plot_metric, plot_decision_boundaries


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config):
    seed = int(config.get("seed", 42))
    set_seed(seed)

    output_dir = Path(config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y = make_two_moons(
        n_samples=int(config.get("n_samples", 1200)),
        noise=float(config.get("noise", 0.16)),
        seed=seed,
    )
    x_train, y_train, x_test, y_test = train_test_split(
        x, y,
        test_fraction=float(config.get("test_fraction", 0.25)),
        seed=seed + 1,
    )

    hidden_dim = int(config.get("hidden_dim", 64))
    epochs = int(config.get("epochs", 150))
    learning_rates = config.get("learning_rates", {
        "sgd": 0.08, "momentum": 0.08, "adam": 0.01, "lbfgs": 0.8
    })

    # Establish one common initialization and load it into every model.
    set_seed(seed + 2)
    base_model = MLP(hidden_dim=hidden_dim)
    initial_state = {
        key: value.detach().clone()
        for key, value in base_model.state_dict().items()
    }

    histories = {}
    models = {}
    summary_rows = []
    all_metric_rows = []

    for name, learning_rate in learning_rates.items():
        model = MLP(hidden_dim=hidden_dim)
        model.load_state_dict(initial_state)
        optimizer = build_optimizer(name, model, float(learning_rate))

        history = train_model(model, optimizer, x_train, y_train, epochs)
        test_loss, test_accuracy = evaluate(
            model, x_test, y_test, torch.nn.CrossEntropyLoss()
        )

        histories[name] = history
        models[name] = model

        final = history[-1]
        summary_rows.append({
            "optimizer": name,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "final_train_loss": final["train_loss"],
            "final_train_accuracy": final["train_accuracy"],
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "final_gradient_norm": final["gradient_norm"],
        })

        for row in history:
            all_metric_rows.append({"optimizer": name, **row})

    write_csv(output_dir / "metrics.csv", all_metric_rows)
    write_csv(output_dir / "summary.csv", summary_rows)

    plot_metric(
        histories, "train_loss", "Training cross-entropy loss",
        output_dir / "loss_curves.png",
    )
    plot_metric(
        histories, "gradient_norm", "Parameter gradient L2 norm",
        output_dir / "gradient_norm_curves.png",
    )
    plot_decision_boundaries(
        models, x_train, y_train, output_dir / "decision_boundaries.png"
    )

    metadata = {
        "config": config,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_note": "Record your Python version when reporting results.",
        "protocol": "Full-batch training; common initialization; fixed synthetic dataset.",
    }
    with (output_dir / "experiment_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Experiment complete. Outputs saved to: {output_dir.resolve()}")
    print(f"Summary: { (output_dir / 'summary.csv').resolve() }")
    return summary_rows
