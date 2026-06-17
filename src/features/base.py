from abc import ABC, abstractmethod
from typing import Tuple, Self
import pandas as pd
import numpy as np


class Preprocessor(ABC):
    @abstractmethod
    def apply(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        pass


class FeatureEngineer(ABC):
    def __init__(self):
        self.encoders = {}

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> Self:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        pass
