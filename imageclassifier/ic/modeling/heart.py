"""Functional Keras model for the cardiovascular dataset."""

import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from ic.config import HeartConfig
from ic.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_heart_classifier(
    train_frame: pd.DataFrame,
    config: HeartConfig,
) -> tuple[keras.Model, dict[str, list[str]]]:
    """Build and compile the named-input cardiovascular classifier."""
    vocabularies = {
        name: sorted(train_frame[name].astype(str).unique().tolist())
        for name in CATEGORICAL_FEATURES
    }
    inputs: dict[str, keras.KerasTensor] = {}
    encoded: list[keras.KerasTensor] = []

    for name in NUMERIC_FEATURES:
        feature_input = keras.Input(shape=(1,), name=name, dtype="float32")
        normalizer = layers.Normalization(axis=-1, name=f"{name}_normalization")
        normalizer.adapt(train_frame[name].to_numpy(dtype="float32").reshape(-1, 1))
        inputs[name] = feature_input
        encoded.append(normalizer(feature_input))

    for name in CATEGORICAL_FEATURES:
        feature_input = keras.Input(shape=(1,), name=name, dtype="int32")
        lookup = layers.IntegerLookup(
            vocabulary=list(range(len(vocabularies[name]))),
            num_oov_indices=1,
            output_mode="one_hot",
            name=f"{name}_lookup",
        )
        inputs[name] = feature_input
        encoded.append(lookup(feature_input))

    features = layers.Concatenate(name="all_features")(encoded)
    features = layers.Dense(64, activation="relu", name="dense_1")(features)
    features = layers.Dropout(0.30, name="dropout")(features)
    features = layers.Dense(32, activation="relu", name="dense_2")(features)
    output = layers.Dense(1, activation="sigmoid", name="probability")(features)
    model = keras.Model(inputs=inputs, outputs=output, name="heart_disease_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model, vocabularies
