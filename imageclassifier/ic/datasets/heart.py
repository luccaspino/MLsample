"""Heart disease dataset loading and deterministic splitting."""

from dataclasses import dataclass

import pandas as pd

from ic.config import HeartConfig


@dataclass(frozen=True)
class HeartData:
    train: pd.DataFrame
    validation: pd.DataFrame


def load_heart_data(config: HeartConfig, path: str | None = None) -> pd.DataFrame:
    """Load heart data from a local path or the configured URL."""
    source = path or config.data_url
    dataframe = pd.read_csv(source)
    if "target" not in dataframe.columns:
        raise ValueError("Heart dataset must contain a 'target' column")
    return dataframe


def split_heart_data(dataframe: pd.DataFrame, config: HeartConfig) -> HeartData:
    """Create a reproducible, stratified train/validation split."""
    validation_parts: list[pd.DataFrame] = []
    train_parts: list[pd.DataFrame] = []
    for _, group in dataframe.groupby("target", sort=False):
        validation_part = group.sample(
            frac=config.validation_fraction,
            random_state=config.seed,
        )
        validation_parts.append(validation_part)
        train_parts.append(group.drop(validation_part.index))

    validation = pd.concat(validation_parts).sample(frac=1, random_state=config.seed)
    train = pd.concat(train_parts).sample(frac=1, random_state=config.seed)
    return HeartData(train=train, validation=validation)
