"""Visualization utilities for optimizer comparison results."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(history, output_path):
    """Plot training loss over epochs."""

    plt.figure(figsize=(8, 5))

    for name, losses in history.items():
        plt.plot(losses, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Optimizer Training Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_accuracy_comparison(results, output_path):
    """Plot test accuracy for different optimizers."""

    df = pd.DataFrame(results)

    plt.figure(figsize=(8, 5))
    plt.bar(df["optimizer"], df["test_accuracy"])
    plt.xlabel("Optimizer")
    plt.ylabel("Test accuracy")
    plt.title("Test Accuracy Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
