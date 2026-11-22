from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_metric(histories, metric, ylabel, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        values = [row[metric] for row in history]
        ax.plot(epochs, values, label=name.upper())
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} during training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_decision_boundaries(models, x_train, y_train, output_path):
    x_np = x_train.detach().cpu().numpy()
    y_np = y_train.detach().cpu().numpy()
    x_min, x_max = x_np[:, 0].min() - 0.5, x_np[:, 0].max() + 0.5
    y_min, y_max = x_np[:, 1].min() - 0.5, x_np[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 220),
        np.linspace(y_min, y_max, 220),
    )
    grid = np.column_stack((xx.ravel(), yy.ravel()))

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for ax, (name, model) in zip(axes[0], models.items()):
        import torch
        model.eval()
        with torch.no_grad():
            predictions = model(torch.tensor(grid, dtype=torch.float32)).argmax(dim=1)
        zz = predictions.cpu().numpy().reshape(xx.shape)
        ax.contourf(xx, yy, zz, alpha=0.25, levels=[-0.5, 0.5, 1.5])
        ax.scatter(x_np[:, 0], x_np[:, 1], c=y_np, s=9, edgecolors="k", linewidths=0.15)
        ax.set_title(name.upper())
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
