from .base import FeatureEngineer
from typing import Tuple, Self
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class MLPFeatureEngineer(FeatureEngineer):
    """
    Feature engineer for MLP model that creates features from ratings data.

    Features engineered:
    1. User statistics (mean, std, count of ratings)
    2. Movie statistics (mean, std, count of ratings)
    3. Genre one-hot encoding
    4. Time-based features (hour, day of week)
    5. Interaction features (user-movie average rating interaction)

    All numerical features are standardized using StandardScaler.
    """

    def __init__(self):
        super().__init__()
        self.feature_columns = []

    def fit(self, df: pd.DataFrame) -> Self:
        """
        Fit the feature engineer on training data.

        Computes user statistics, movie statistics, identifies genre columns,
        and fits the StandardScaler on engineered features.

        Args:
            df: DataFrame containing ratings data merged with movie genres
                Required columns: userId, movieId, rating, timestamp, genres

        Returns:
            Self: Fitted feature engineer instance
        """
        # Create a copy to avoid modifying original
        df_copy = df.copy()

        # Feature 1: User statistics
        user_stats = (
            df_copy.groupby("userId")
            .agg({"rating": ["mean", "std", "count"]})
            .fillna(0)
        )
        user_stats.columns = ["user_avg_rating", "user_std_rating", "user_rating_count"]

        # Store user stats for transform
        self.encoders["user_stats"] = user_stats

        # Store global statistics for unseen users/movies
        self.encoders["global_user_stats"] = {
            "user_avg_rating": df_copy["rating"].mean(),
            "user_std_rating": 0,  # No std for single user
            "user_rating_count": 1,  # Minimum count
        }

        # Feature 2: Movie statistics
        movie_stats = (
            df_copy.groupby("movieId")
            .agg({"rating": ["mean", "std", "count"]})
            .fillna(0)
        )
        movie_stats.columns = [
            "movie_avg_rating",
            "movie_std_rating",
            "movie_rating_count",
        ]

        # Store movie stats for transform
        self.encoders["movie_stats"] = movie_stats

        # Store global statistics for unseen movies
        self.encoders["global_movie_stats"] = {
            "movie_avg_rating": df_copy["rating"].mean(),
            "movie_std_rating": 0,  # No std for single movie
            "movie_rating_count": 1,  # Minimum count
        }

        # Apply features to training data to identify all columns
        df_features = self._create_features(df_copy, user_stats, movie_stats)

        # Store genre columns (they're identified during feature creation)
        genre_columns = [
            col
            for col in df_features.columns
            if col
            not in [
                "userId",
                "movieId",
                "rating",
                "timestamp",
                "genres",
                "rating_timestamp",
                "user_avg_rating",
                "user_std_rating",
                "user_rating_count",
                "movie_avg_rating",
                "movie_std_rating",
                "movie_rating_count",
                "rating_hour",
                "rating_dayofweek",
                "user_movie_avg_interaction",
            ]
        ]

        # Feature 5: Interaction features
        df_features["user_movie_avg_interaction"] = (
            df_features["user_avg_rating"] * df_features["movie_avg_rating"]
        )

        # Select feature columns (exclude non-feature columns)
        self.feature_columns = [
            col
            for col in df_features.columns
            if col
            not in [
                "userId",
                "movieId",
                "rating",
                "timestamp",
                "genres",
                "rating_timestamp",
            ]
        ]

        # Store genre columns for transform
        self.encoders["genre_columns"] = genre_columns

        # Fit scaler on training features
        X = df_features[self.feature_columns]
        self.encoders["scaler"] = StandardScaler()
        self.encoders["scaler"].fit(X)

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """
        Transform data using fitted feature engineer.

        Args:
            df: DataFrame containing ratings data merged with movie genres
                Required columns: userId, movieId, rating, timestamp, genres

        Returns:
            Tuple containing:
                - X_scaled: Scaled feature matrix (np.ndarray)
                - y: Target ratings (pd.Series)

        Raises:
            ValueError: If feature engineer hasn't been fitted yet
        """
        if "scaler" not in self.encoders:
            raise ValueError("Feature engineer must be fitted before transform")

        df_copy = df.copy()

        # Apply stored statistics with handling for unseen users/movies
        user_stats = self.encoders["user_stats"]
        movie_stats = self.encoders["movie_stats"]

        # Create features (this will introduce NaN for unseen users/movies)
        df_features = self._create_features(df_copy, user_stats, movie_stats)

        # Fill NaN values for unseen users
        global_user = self.encoders["global_user_stats"]
        df_features["user_avg_rating"] = df_features["user_avg_rating"].fillna(
            global_user["user_avg_rating"]
        )
        df_features["user_std_rating"] = df_features["user_std_rating"].fillna(
            global_user["user_std_rating"]
        )
        df_features["user_rating_count"] = df_features["user_rating_count"].fillna(
            global_user["user_rating_count"]
        )

        # Fill NaN values for unseen movies
        global_movie = self.encoders["global_movie_stats"]
        df_features["movie_avg_rating"] = df_features["movie_avg_rating"].fillna(
            global_movie["movie_avg_rating"]
        )
        df_features["movie_std_rating"] = df_features["movie_std_rating"].fillna(
            global_movie["movie_std_rating"]
        )
        df_features["movie_rating_count"] = df_features["movie_rating_count"].fillna(
            global_movie["movie_rating_count"]
        )

        # Feature 5: Interaction features
        df_features["user_movie_avg_interaction"] = (
            df_features["user_avg_rating"] * df_features["movie_avg_rating"]
        )

        # Ensure all expected columns exist (handle missing genres in test data)
        for col in self.feature_columns:
            if col not in df_features.columns:
                df_features[col] = 0

        # Select and order feature columns
        X = df_features[self.feature_columns]
        y = df_features["rating"]

        # Double-check no NaN values remain
        if X.isnull().any().any():
            # Fill any remaining NaN with 0 as a last resort
            X = X.fillna(0)

        # Scale features
        X_scaled = self.encoders["scaler"].transform(X)

        return X_scaled, y

    def _create_features(
        self, df: pd.DataFrame, user_stats: pd.DataFrame, movie_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create all features from the dataframe.

        Args:
            df: Input dataframe
            user_stats: Pre-computed user statistics
            movie_stats: Pre-computed movie statistics

        Returns:
            DataFrame with all features added
        """
        df = df.copy()

        # Feature 1: Merge user statistics (use left join to keep all rows)
        df = df.merge(user_stats, on="userId", how="left")

        # Feature 2: Merge movie statistics (use left join to keep all rows)
        df = df.merge(movie_stats, on="movieId", how="left")

        # Feature 3: Genre one-hot encoding
        genres_dummies = df["genres"].str.get_dummies(sep="|")
        df = pd.concat([df, genres_dummies], axis=1)

        # Feature 4: Time-based features
        df["rating_timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["rating_hour"] = df["rating_timestamp"].dt.hour
        df["rating_dayofweek"] = df["rating_timestamp"].dt.dayofweek

        return df
