"""Tests for the deep learning optimization project."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dlopt.data import load_dataset


def test_dataset_shapes():
    x_train, x_test, y_train, y_test = load_dataset()

    assert x_train.shape[1] == 2
    assert x_test.shape[1] == 2
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)


def test_dataset_is_not_empty():
    x_train, x_test, y_train, y_test = load_dataset()

    assert len(x_train) > 0
    assert len(x_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0


def test_dataset_contains_two_classes():
    x_train, x_test, y_train, y_test = load_dataset()

    labels = np.unique(np.concatenate((y_train, y_test)))

    assert len(labels) == 2
