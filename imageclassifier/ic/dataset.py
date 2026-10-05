"""Dataset loading and persistence functions exposed by the package."""

from pathlib import Path

import pandas as pd

from ic.datasets.cifar10 import Cifar10Data, get_class_names, load_cifar10
from ic.datasets.delivery import make_delivery_dataset
from ic.datasets.heart import HeartData, load_heart_data, split_heart_data


def save_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    """Save a tabular dataset, creating its destination directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)


__all__ = [
    "Cifar10Data",
    "HeartData",
    "get_class_names",
    "load_cifar10",
    "load_heart_data",
    "make_delivery_dataset",
    "save_dataset",
    "split_heart_data",
]
