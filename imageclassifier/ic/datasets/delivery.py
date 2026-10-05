"""Synthetic delivery dataset used by the TensorFlow lesson."""

import tensorflow as tf


def make_delivery_dataset() -> tuple[tf.Tensor, tf.Tensor]:
    """Return the six normalized delivery examples and binary targets."""
    features = tf.constant(
        [
            [0.20, 0.10, 0.30],
            [0.80, 0.70, 0.90],
            [0.30, 0.20, 0.40],
            [0.90, 0.80, 0.70],
            [0.15, 0.30, 0.20],
            [0.75, 0.60, 0.85],
        ],
        dtype=tf.float32,
    )
    targets = tf.constant([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]], dtype=tf.float32)
    return features, targets
