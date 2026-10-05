"""CIFAR-10 loading utilities."""

from dataclasses import dataclass

import numpy as np
from tensorflow import keras


@dataclass(frozen=True)
class Cifar10Data:
    """Normalized CIFAR-10 arrays."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


CLASS_NAMES = (
    "avião",
    "automóvel",
    "pássaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhão",
)


def load_cifar10() -> Cifar10Data:
    """Download CIFAR-10 through Keras and normalize image pixels."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return Cifar10Data(
        x_train=x_train.astype("float32") / 255.0,
        y_train=y_train.reshape(-1).astype("int32"),
        x_test=x_test.astype("float32") / 255.0,
        y_test=y_test.reshape(-1).astype("int32"),
    )


def get_class_names() -> tuple[str, ...]:
    """Return the canonical CIFAR-10 class names."""
    return CLASS_NAMES
