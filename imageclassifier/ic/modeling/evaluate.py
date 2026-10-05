"""Evaluation and thresholding helpers."""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras


@dataclass(frozen=True)
class BinaryMetrics:
    """Confusion counts for binary predictions."""

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


def probabilities_to_classes(
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert probabilities into binary class labels."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return (np.asarray(probabilities) >= threshold).astype("int32")


def binary_metrics(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> BinaryMetrics:
    """Calculate binary confusion counts."""
    true_values = np.asarray(true_values).reshape(-1)
    predicted_values = np.asarray(predicted_values).reshape(-1)
    return BinaryMetrics(
        true_negative=int(np.sum((predicted_values == 0) & (true_values == 0))),
        false_positive=int(np.sum((predicted_values == 1) & (true_values == 0))),
        false_negative=int(np.sum((predicted_values == 0) & (true_values == 1))),
        true_positive=int(np.sum((predicted_values == 1) & (true_values == 1))),
    )


def evaluate_classifier(
    model: keras.Model,
    features,
    targets,
) -> dict[str, float]:
    """Evaluate a compiled Keras model and return scalar metrics."""
    results = model.evaluate(features, targets, return_dict=True, verbose=0)
    return {name: float(value) for name, value in results.items()}


def binary_cross_entropy(
    true_values: tf.Tensor,
    probabilities: tf.Tensor,
) -> float:
    """Calculate mean binary cross-entropy for teaching examples."""
    loss = keras.losses.BinaryCrossentropy()(true_values, probabilities)
    return float(loss.numpy())
