"""Training and artifact persistence workflows."""

import json
from pathlib import Path

from tensorflow import keras

from ic.config import Cifar10Config, HeartConfig
from ic.datasets.cifar10 import Cifar10Data
from ic.features import make_heart_inputs


def train_image_classifier(
    model: keras.Model,
    data: Cifar10Data,
    config: Cifar10Config,
    checkpoint_path: Path | None = None,
) -> keras.callbacks.History:
    """Train the CIFAR-10 model and optionally save the best checkpoint."""
    callbacks: list[keras.callbacks.Callback] = []
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
            )
        )
    return model.fit(
        data.x_train,
        data.y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        callbacks=callbacks,
        verbose=1,
    )


def save_metadata(
    metadata_path: Path,
    model_name: str,
    threshold: float,
    selection_metric: str,
) -> None:
    """Persist model metadata next to the serialized model."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "threshold": float(threshold),
                "selection_metric": selection_metric,
            },
            indent=4,
        ),
        encoding="utf-8",
    )


def train_heart_classifier(
    model: keras.Model,
    train_frame,
    validation_frame,
    vocabularies: dict[str, list[str]],
    config: HeartConfig,
    checkpoint_path: Path | None = None,
) -> keras.callbacks.History:
    """Train the cardiovascular model with validation and early stopping."""
    train_inputs = make_heart_inputs(train_frame, vocabularies)
    validation_inputs = make_heart_inputs(validation_frame, vocabularies)
    y_train = train_frame["target"].to_numpy(dtype="float32").reshape(-1, 1)
    y_validation = validation_frame["target"].to_numpy(dtype="float32").reshape(-1, 1)
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=12,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        ),
    ]
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_auc",
                mode="max",
                save_best_only=True,
            )
        )
    return model.fit(
        train_inputs,
        y_train,
        validation_data=(validation_inputs, y_validation),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
