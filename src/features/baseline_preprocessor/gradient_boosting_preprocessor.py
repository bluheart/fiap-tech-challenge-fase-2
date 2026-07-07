from ..base import Preprocessor
import pandas as pd


class GradientBoostingPreprocessor(Preprocessor):
    """
    Preprocessor for Gradient Boosting models that filters and merges data.

    This preprocessor removes movies with no genres listed and enriches the
    ratings dataset with genre information from the movies dataset.
    """

    def apply(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing to the ratings and movies dataframes.

        Steps performed:
        1. Filters out movies with genre "(no genres listed)"
        2. Merges ratings with movie genres on movieId

        Args:
            ratings_df: DataFrame containing user ratings with columns including 'movieId'
            movies_df: DataFrame containing movie metadata with columns including 'movieId' and 'genres'

        Returns:
            pd.DataFrame: Preprocessed DataFrame containing ratings merged with movie genres,
                         excluding movies with no genres listed

        Raises:
            KeyError: If required columns ('movieId' or 'genres') are missing from input DataFrames
        """
        # Filter out movies with no genres
        movies_df = movies_df[movies_df["genres"] != "(no genres listed)"]

        # Create copy to avoid modifying original
        df = ratings_df.copy()
        # Merge with movies to get genres
        df = df.merge(movies_df[["movieId", "genres"]], on="movieId", how="inner")
        return df
