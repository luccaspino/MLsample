"""Tests for thresholding and binary evaluation helpers."""

import numpy as np
import tensorflow as tf

from ic.modeling.evaluate import binary_cross_entropy, binary_metrics, probabilities_to_classes


def test_probabilities_to_classes_uses_threshold():
    result = probabilities_to_classes(np.array([0.49, 0.5, 0.9]), threshold=0.5)

    assert result.tolist() == [0, 1, 1]


def test_binary_metrics_counts_confusion_matrix():
    result = binary_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))

    assert result.true_negative == 1
    assert result.false_positive == 1
    assert result.false_negative == 1
    assert result.true_positive == 1


def test_binary_cross_entropy_penalizes_confident_wrong_prediction():
    correct = binary_cross_entropy(tf.constant([[1.0]]), tf.constant([[0.95]]))
    wrong = binary_cross_entropy(tf.constant([[1.0]]), tf.constant([[0.05]]))

    assert correct < wrong
