"""Inference and serialized model loading helpers."""

import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from ic.features import make_heart_inputs


def load_model(model_path: Path, metadata_path: Path) -> tuple[keras.Model, str, float]:
    """Load a Keras model and its inference metadata."""
    model = keras.models.load_model(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, str(metadata["model_name"]), float(metadata["threshold"])


def predict_images(model: keras.Model, images: np.ndarray) -> np.ndarray:
    """Return class probabilities for CIFAR-10 images."""
    return model.predict(images, verbose=0)


def predict_heart_patient(
    model: keras.Model,
    patient: dict[str, object],
    vocabularies: dict[str, list[str]],
) -> float:
    """Return the cardiovascular probability for one patient."""
    import pandas as pd

    inputs = make_heart_inputs(pd.DataFrame([patient]), vocabularies)
    return float(model.predict(inputs, verbose=0)[0, 0])
