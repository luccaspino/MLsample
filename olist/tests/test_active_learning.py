import pandas as pd
from sklearn.linear_model import LogisticRegression

from module_olist.modeling.active_learning import train_with_uncertainty_sampling


def test_active_learning_queries_until_label_budget():
    X = pd.DataFrame({"value": range(12)})
    y = pd.Series([0, 1] * 6)

    result = train_with_uncertainty_sampling(
        estimator=LogisticRegression(),
        X=X,
        y=y,
        initial_size=4,
        batch_size=3,
        max_labeled=10,
        random_state=7,
    )

    assert len(result.labeled_indices) == 10
    assert sum(round_info["queried"] for round_info in result.query_history) == 10
    assert result.model.predict_proba(X).shape == (12, 2)


def test_active_learning_starts_with_all_classes():
    X = pd.DataFrame({"value": range(6)})
    y = pd.Series([0, 0, 0, 1, 1, 1])

    result = train_with_uncertainty_sampling(
        estimator=LogisticRegression(),
        X=X,
        y=y,
        initial_size=2,
        batch_size=2,
        max_labeled=2,
        random_state=3,
    )

    assert set(y.loc[result.labeled_indices]) == {0, 1}