from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone


@dataclass
class ActiveLearningResult:
    model: object
    labeled_indices: list
    query_history: list


def _select_initial_indices(y: pd.Series, initial_size: int, random_state: int) -> list:
    if initial_size < y.nunique():
        raise ValueError("initial_size deve incluir pelo menos uma amostra por classe")

    rng = np.random.default_rng(random_state)
    positions = []
    for class_value in y.unique():
        class_positions = np.flatnonzero(y.to_numpy() == class_value)
        positions.append(int(rng.choice(class_positions)))

    remaining = np.setdiff1d(np.arange(len(y)), positions)
    extra_size = initial_size - len(positions)
    if extra_size:
        positions.extend(rng.choice(remaining, size=extra_size, replace=False).tolist())

    return positions


def train_with_uncertainty_sampling(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    initial_size: int = 100,
    batch_size: int = 50,
    max_labeled: int | None = None,
    random_state: int = 42,
) -> ActiveLearningResult:
    """Treina um estimador usando active learning por amostragem de incerteza.

    O conjunto ``y`` representa o oraculo de rotulos disponivel durante a
    simulacao. Em um fluxo real, os indices consultados seriam enviados para
    anotacao humana antes da proxima rodada.
    """
    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo numero de amostras")
    if len(X) == 0:
        raise ValueError("X e y nao podem estar vazios")
    if initial_size < 1 or initial_size > len(X):
        raise ValueError("initial_size deve estar entre 1 e o tamanho de X")
    if batch_size < 1:
        raise ValueError("batch_size deve ser positivo")

    target_size = len(X) if max_labeled is None else min(max_labeled, len(X))
    if target_size < initial_size:
        raise ValueError("max_labeled deve ser maior ou igual a initial_size")

    labeled_positions = _select_initial_indices(y, initial_size, random_state)
    unlabeled_positions = [
        position for position in range(len(X)) if position not in labeled_positions
    ]
    query_history = [{"round": 0, "queried": len(labeled_positions)}]
    model = clone(estimator)

    while True:
        model.fit(X.iloc[labeled_positions], y.iloc[labeled_positions])
        if not unlabeled_positions or len(labeled_positions) >= target_size:
            break

        probabilities = model.predict_proba(X.iloc[unlabeled_positions])[:, 1]
        uncertainty = np.abs(probabilities - 0.5)
        query_size = min(batch_size, target_size - len(labeled_positions), len(unlabeled_positions))
        selected_order = np.argsort(uncertainty)[:query_size]
        selected_positions = [unlabeled_positions[index] for index in selected_order]

        labeled_positions.extend(selected_positions)
        selected_set = set(selected_positions)
        unlabeled_positions = [
            position for position in unlabeled_positions if position not in selected_set
        ]
        query_history.append(
            {"round": len(query_history), "queried": len(selected_positions)}
        )

    return ActiveLearningResult(
        model=model,
        labeled_indices=X.index[labeled_positions].tolist(),
        query_history=query_history,
    )