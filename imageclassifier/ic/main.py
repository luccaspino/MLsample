"""Typer commands for reproducible imageclassifier workflows."""

from pathlib import Path
from typing import Annotated

import typer

from ic.config import CHECKPOINTS_DIR, MODELS_DIR, Cifar10Config, HeartConfig
from ic.dataset import load_cifar10, load_heart_data
from ic.datasets.heart import split_heart_data
from ic.modeling.cnn import build_cifar10_cnn
from ic.modeling.heart import build_heart_classifier
from ic.modeling.train import save_metadata, train_heart_classifier, train_image_classifier

app = typer.Typer(help="Machine learning workflows for imageclassifier.")


@app.command("train-cifar10")
def train_cifar10() -> None:
    """Train and save the CIFAR-10 CNN."""
    config = Cifar10Config()
    data = load_cifar10()
    model = build_cifar10_cnn(config)
    checkpoint_path = CHECKPOINTS_DIR / config.model_name
    train_image_classifier(model, data, config, checkpoint_path)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / config.model_name)
    save_metadata(
        MODELS_DIR / "cifar10_metadata.json",
        config.model_name,
        threshold=0.5,
        selection_metric="validation_accuracy",
    )


@app.command("train-heart")
def train_heart(
    path: Annotated[Path | None, typer.Option(help="Local heart.csv path")] = None,
) -> None:
    """Train and save the cardiovascular classifier."""
    config = HeartConfig()
    data = split_heart_data(load_heart_data(config, str(path) if path else None), config)
    model, vocabularies = build_heart_classifier(data.train, config)
    checkpoint_path = CHECKPOINTS_DIR / config.model_name
    train_heart_classifier(
        model, data.train, data.validation, vocabularies, config, checkpoint_path
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / config.model_name)
    save_metadata(
        MODELS_DIR / "heart_metadata.json",
        config.model_name,
        config.threshold,
        selection_metric="validation_auc",
    )


if __name__ == "__main__":
    app()
