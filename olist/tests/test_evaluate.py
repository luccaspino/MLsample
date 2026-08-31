import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from module_olist.modeling.evaluate import cross_validate_models, evaluate_models


def test_cross_validation_selects_threshold_without_fitting_original_model():
    x_train = pd.DataFrame({"feature": np.arange(30)})
    y_train = pd.Series([0, 0, 1] * 10)
    models = {"LogisticRegression": LogisticRegression()}

    results = cross_validate_models(models, x_train, y_train, n_splits=3)

    assert set(results["LogisticRegression"]) == {
        "threshold",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }
    assert 0.05 <= results["LogisticRegression"]["threshold"] < 0.5
    assert not hasattr(models["LogisticRegression"], "classes_")


def test_evaluate_models_uses_provided_threshold():
    x_train = pd.DataFrame({"feature": np.arange(20)})
    y_train = pd.Series([0, 1] * 10)
    model = LogisticRegression().fit(x_train, y_train)

    results = evaluate_models(
        {"LogisticRegression": model},
        x_train,
        y_train,
        {"LogisticRegression": 0.25},
    )

    result = results["LogisticRegression"]
    assert result["threshold"] == 0.25
    assert result["tn"] + result["fp"] + result["fn"] + result["tp"] == len(y_train)
