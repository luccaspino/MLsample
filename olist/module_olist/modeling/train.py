import pandas as pd

from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_lightgbm_pipeline,
    create_xgboost_pipeline,
)


def create_models():
    return {
        "GradientBoosting": create_gradient_boosting_pipeline(),
        "XGBoost": create_xgboost_pipeline(),
        "LightGBM": create_lightgbm_pipeline(),
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models=None,
):
    if models is None:
        models = create_models()

    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        print(f"{name} training completed.")
        trained_models[name] = model

    return trained_models
