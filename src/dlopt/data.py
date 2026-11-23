"""Dataset generation and preprocessing utilities."""

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


def load_dataset(
    n_samples=1200,
    noise=0.20,
    test_size=0.25,
    random_state=42,
):
    """Generate a two-moons dataset and split it without data leakage."""

    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
    )
