"""Tests for dataset and feature preparation contracts."""

import numpy as np
import pandas as pd

from ic.dataset import make_delivery_dataset
from ic.datasets.heart import split_heart_data
from ic.config import HeartConfig


def test_delivery_dataset_has_expected_shapes():
    features, targets = make_delivery_dataset()

    assert tuple(features.shape) == (6, 3)
    assert tuple(targets.shape) == (6, 1)


def test_heart_split_preserves_target_distribution():
    data = pd.DataFrame({"target": [0] * 10 + [1] * 10})
    result = split_heart_data(data, HeartConfig(validation_fraction=0.2))

    assert len(result.train) == 16
    assert len(result.validation) == 4
    assert set(result.train["target"]) == {0, 1}
    assert set(result.validation["target"]) == {0, 1}
