"""Small TensorFlow models and operations used in the teaching notebook."""

import tensorflow as tf
from tensorflow import keras


def linear_neuron(
    features: tf.Tensor,
    weights: tf.Tensor,
    bias: tf.Tensor,
) -> tf.Tensor:
    """Calculate a neuron's linear output."""
    return tf.reduce_sum(features * weights) + bias


def apply_relu(values: tf.Tensor) -> tf.Tensor:
    """Apply the ReLU activation."""
    return tf.nn.relu(values)


def build_delivery_mlp() -> keras.Model:
    """Build the untrained 3 -> 4 -> 1 lesson model."""
    return keras.Sequential(
        [
            keras.Input(shape=(3,)),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="delivery_mlp",
    )


def count_parameters(model: keras.Model) -> int:
    """Return the number of trainable and non-trainable parameters."""
    return int(model.count_params())
