import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import module_olist.modeling.train as train_module
from module_olist.modeling.evaluate import evaluate_model
from module_olist.modeling.train import create_selected_model, train_model


def test_evaluate_model_uses_provided_threshold():
    x_train = pd.DataFrame({"feature": np.arange(20)})
    y_train = pd.Series([0, 1] * 10)
    model = LogisticRegression().fit(x_train, y_train)

    result = evaluate_model(
        model=model,
        model_name="LogisticRegression",
        X_test=x_train,
        y_test=y_train,
        threshold=0.25,
    )

    assert result is None


def test_create_selected_model_rejects_unknown_name():
    try:
        create_selected_model("Unknown")
    except ValueError as error:
        assert "Modelo desconhecido" in str(error)
    else:
        raise AssertionError("Era esperado ValueError para um modelo desconhecido")


def test_train_model_saves_pipeline_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        train_module,
        "create_selected_model",
        lambda model_name: LogisticRegression(),
    )
    x_train = pd.DataFrame({"feature": np.arange(20)})
    y_train = pd.Series([0, 1] * 10)
    model_path = tmp_path / "best_model.joblib"
    metadata_path = tmp_path / "metadata.json"

    model = train_model(
        model_name="LogisticRegression",
        threshold=0.25,
        X_train=x_train,
        y_train=y_train,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    saved_model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert isinstance(model, LogisticRegression)
    assert isinstance(saved_model, LogisticRegression)
    assert metadata == {
        "model_name": "LogisticRegression",
        "threshold": 0.25,
        "selection_metric": "pr_auc",
        "threshold_metric": "f1",
    }
