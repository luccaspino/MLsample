"""Convolutional model for CIFAR-10."""

from tensorflow import keras
from tensorflow.keras import layers

from ic.config import Cifar10Config


def build_cifar10_cnn(config: Cifar10Config) -> keras.Model:
    """Build the CNN used by the image classification workflow."""
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="cifar10_cnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
