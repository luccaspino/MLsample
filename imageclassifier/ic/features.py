"""Feature preparation functions for the cardiovascular workflow."""

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "slope"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "ca", "thal"]


def make_heart_inputs(
    frame: pd.DataFrame,
    categorical_vocabularies: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Convert a heart dataframe into named model inputs."""
    inputs: dict[str, np.ndarray] = {}
    for name in NUMERIC_FEATURES:
        inputs[name] = frame[name].to_numpy(dtype="float32").reshape(-1, 1)
    for name in CATEGORICAL_FEATURES:
        mapping = {value: index for index, value in enumerate(categorical_vocabularies[name])}
        values = frame[name].astype(str).map(mapping).fillna(-1)
        inputs[name] = values.to_numpy(dtype="int32").reshape(-1, 1)
    return inputs


__all__ = [
    "CATEGORICAL_FEATURES",
    "NUMERIC_FEATURES",
    "make_heart_inputs",
]
