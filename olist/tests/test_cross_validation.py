import numpy as np
import pandas as pd

import module_olist.modeling.cross_validation as cross_validation


class Candidate:
    def __init__(self, name):
        self.name = name


def test_cross_validation_selects_model_by_pr_auc(monkeypatch):
    monkeypatch.setattr(
        cross_validation,
        "create_gradient_boosting_pipeline",
        lambda: Candidate("Gradient Boosting"),
    )
    monkeypatch.setattr(
        cross_validation,
        "create_xgboost_pipeline",
        lambda: Candidate("XGBoost"),
    )
    monkeypatch.setattr(
        cross_validation,
        "create_lightgbm_pipeline",
        lambda: Candidate("LightGBM"),
    )

    pr_auc_by_model = {
        "Gradient Boosting": 0.80,
        "XGBoost": 0.70,
        "LightGBM": 0.60,
    }

    def fake_cross_validate(estimator, **kwargs):
        results = {}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            results[f"test_{metric}"] = np.full(5, 0.50)
        results["test_pr_auc"] = np.full(5, pr_auc_by_model[estimator.name])
        return results

    def fake_cross_val_predict(estimator, X, **kwargs):
        probabilities = np.where(X["target_probability"].to_numpy() == 1, 0.9, 0.1)
        return np.column_stack((1 - probabilities, probabilities))

    monkeypatch.setattr(cross_validation, "cross_validate", fake_cross_validate)
    monkeypatch.setattr(cross_validation, "cross_val_predict", fake_cross_val_predict)

    X_train = pd.DataFrame({"target_probability": [0, 1] * 5})
    y_train = pd.Series([0, 1] * 5)

    model_name, threshold = cross_validation.cross_validate_models(X_train, y_train)

    assert model_name == "Gradient Boosting"
    assert threshold == 0.11
