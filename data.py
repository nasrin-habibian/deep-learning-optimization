import numpy as np
import torch


def make_two_moons(n_samples=1200, noise=0.16, seed=42):
    """Generate a noisy two-moons dataset without a scikit-learn dependency."""
    if n_samples < 20:
        raise ValueError("n_samples must be at least 20.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")

    rng = np.random.default_rng(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    theta_outer = rng.uniform(0.0, np.pi, n_outer)
    outer = np.column_stack((np.cos(theta_outer), np.sin(theta_outer)))

    theta_inner = rng.uniform(0.0, np.pi, n_inner)
    inner = np.column_stack((1.0 - np.cos(theta_inner), 0.5 - np.sin(theta_inner)))

    x = np.vstack((outer, inner))
    y = np.concatenate((np.zeros(n_outer), np.ones(n_inner))).astype(np.int64)

    x += rng.normal(0.0, noise, size=x.shape)

    # Shuffle deterministically.
    order = rng.permutation(n_samples)
    x, y = x[order], y[order]

    # Standardize using the whole generated dataset for a simple controlled benchmark.
    # For real-world predictive modelling, fit preprocessing on training data only.
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = (x - mean) / np.maximum(std, 1e-8)

    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def train_test_split(x, y, test_fraction=0.25, seed=123):
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(x), generator=generator)
    n_test = max(1, int(round(len(x) * test_fraction)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]
