"""Plots shared by notebooks and command-line workflows."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ic.datasets.cifar10 import get_class_names


def plot_training_history(history, output_path: Path | None = None) -> None:
    """Plot accuracy and loss from a Keras history object."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Treino")
    axes[0].plot(history.history["val_accuracy"], label="Validação")
    axes[0].set_title("Acurácia")
    axes[1].plot(history.history["loss"], label="Treino")
    axes[1].plot(history.history["val_loss"], label="Validação")
    axes[1].set_title("Loss")
    for axis in axes:
        axis.set_xlabel("Época")
        axis.legend()
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
    plt.show()


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: tuple[str, ...] | None = None,
    output_path: Path | None = None,
) -> None:
    """Render a confusion matrix with class labels."""
    labels = labels or get_class_names()
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Classe prevista")
    axis.set_ylabel("Classe real")
    threshold = matrix.max() / 2
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
    plt.show()
