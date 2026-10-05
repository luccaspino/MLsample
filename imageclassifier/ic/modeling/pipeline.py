"""Model factories exposed by the modeling package."""

from ic.config import Cifar10Config, HeartConfig
from ic.modeling.cnn import build_cifar10_cnn
from ic.modeling.heart import build_heart_classifier
from ic.modeling.mlp import build_delivery_mlp

__all__ = [
    "build_cifar10_cnn",
    "build_delivery_mlp",
    "build_heart_classifier",
]


def create_cifar10_pipeline(config: Cifar10Config):
    """Create the compiled CIFAR-10 model."""
    return build_cifar10_cnn(config)


def create_heart_pipeline(train_frame, config: HeartConfig):
    """Create the compiled cardiovascular model and vocabularies."""
    return build_heart_classifier(train_frame, config)
