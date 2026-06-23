from ..base import Preprocessor
import pandas as pd


class LinearRegressionPreprocessor(Preprocessor):
    def apply(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        # Filter out movies with no genres
        movies_df = movies_df[movies_df["genres"] != "(no genres listed)"]
        # Create copy to avoid modifying original
        df = ratings_df.copy()
        # Merge with movies to get genres
        df = df.merge(movies_df[["movieId", "genres"]], on="movieId", how="inner")
        return df
