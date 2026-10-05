"""Dataset split functions following the project modeling convention."""

import pandas as pd

from ic.config import HeartConfig
from ic.datasets.heart import HeartData, split_heart_data


def split_data(data: pd.DataFrame, config: HeartConfig) -> HeartData:
    """Split cardiovascular data into stratified train and validation frames."""
    return split_heart_data(data, config)


__all__ = ["HeartData", "split_data", "split_heart_data"]
