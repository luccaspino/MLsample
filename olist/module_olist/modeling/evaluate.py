from loguru import logger
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

THRESHOLDS = np.arange(0.05, 0.5, 0.01)


def _calculate_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def _find_best_threshold(y_true, y_proba):
    results = [_calculate_metrics(y_true, y_proba, threshold) for threshold in THRESHOLDS]
    return max(results, key=lambda result: result["f1"])


def cross_validate_models(models, x_train, y_train, n_splits=5):
    """Avalia modelos com previsoes out-of-fold e seleciona thresholds no treino."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    validation_results = {}

    for name, model in models.items():
        logger.info(f"Executando validação cruzada de {n_splits} folds para {name}...")
        y_proba = cross_val_predict(
            model,
            x_train,
            y_train,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        result = _find_best_threshold(y_train, y_proba)
        validation_results[name] = result

        logger.info(
            f"CV - Model: {name} - Best threshold: {result['threshold']:.2f} - "
            f"F1: {result['f1']:.3f} - Precision: {result['precision']:.3f} - "
            f"Recall: {result['recall']:.3f} - ROC AUC: {result['roc_auc']:.3f}"
        )

    return validation_results


def evaluate_models(models, x_test, y_test, thresholds):
    """Avalia o teste uma unica vez usando thresholds definidos no treino."""
    test_results = {}

    for name, model in models.items():
        y_proba = model.predict_proba(x_test)[:, 1]
        threshold = thresholds[name]
        result = _calculate_metrics(y_test, y_proba, threshold)
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        result.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
        test_results[name] = result

        logger.info(
            f"Test - Model: {name} - Threshold: {threshold:.2f} - "
            f"F1: {result['f1']:.3f} - Precision: {result['precision']:.3f} - "
            f"Recall: {result['recall']:.3f} - ROC AUC: {result['roc_auc']:.3f} - "
            f"TN: {tn} - FP: {fp} - FN: {fn} - TP: {tp}"
        )

    return test_results
