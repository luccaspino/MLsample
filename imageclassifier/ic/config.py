"""Project paths and typed configuration for the imageclassifier workflows."""

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used by the project."""

    project_dir: Path = PROJ_ROOT

    @property
    def data_dir(self) -> Path:
        return self.project_dir / DATA_DIR.name

    @property
    def models_dir(self) -> Path:
        return self.project_dir / MODELS_DIR.name

    @property
    def checkpoints_dir(self) -> Path:
        return self.models_dir / CHECKPOINTS_DIR.name

    @property
    def reports_dir(self) -> Path:
        return self.project_dir / REPORTS_DIR.name


@dataclass(frozen=True)
class Cifar10Config:
    """Configuration for the CIFAR-10 convolutional classifier."""

    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    validation_split: float = 0.2
    seed: int = 1337
    model_name: str = "cifar10_cnn.keras"


@dataclass(frozen=True)
class DeliveryConfig:
    """Configuration for the TensorFlow teaching example."""

    threshold: float = 0.5
    seed: int = 1337


@dataclass(frozen=True)
class HeartConfig:
    """Configuration for the cardiovascular tabular classifier."""

    data_url: str = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    threshold: float = 0.5
    seed: int = 1337
    model_name: str = "heart_classifier.keras"


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level configuration shared by scripts and notebooks."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    cifar10: Cifar10Config = field(default_factory=Cifar10Config)
    delivery: DeliveryConfig = field(default_factory=DeliveryConfig)
    heart: HeartConfig = field(default_factory=HeartConfig)
