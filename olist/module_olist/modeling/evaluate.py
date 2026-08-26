from loguru import logger
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def evaluate_models(models, x_test, y_test):
    for name, model in models.items():
        y_proba = model.predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)

        best_threshold = None
        best_f1 = -1
        best_precision = None
        best_recall = None

        for threshold in np.arange(0.05, 0.5, 0.01):
            y_pred = (y_proba >= threshold).astype(int)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
        logger.info(
            f"Model: {name} - Best threshold: {best_threshold:.2f} - "
            f"F1: {best_f1:.2f} - Precision: {best_precision:.2f} - "
            f"Recall: {best_recall:.2f} - ROC AUC: {roc_auc:.2f}"
        )
