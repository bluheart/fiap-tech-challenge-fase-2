from sklearn.decomposition import TruncatedSVD
from ..base import FeatureEngineer
from typing import Self, Tuple
import numpy as np
import pandas as pd


class RandomForestFeatureEng(FeatureEngineer):
    """Feature engineering for Random Forest model.

    This class implements feature engineering specifically designed for random
    forest models, incorporating collaborative filtering via SVD, user and movie
    statistics, genre features, and rating deviations.

    Attributes:
        n_svd_components (int): Number of latent factors for SVD decomposition.
        random_state (int): Random seed for reproducibility.
        encoders (dict): Dictionary storing fitted statistics, SVD model, and
            factor mappings.
    """

    def __init__(self, n_svd_components: int = 20, random_state: int = 42) -> None:
        """Initialize the Random Forest feature engineer.

        Args:
            n_svd_components (int, optional): Number of latent factors to
                extract using Truncated SVD. Defaults to 20.
            random_state (int, optional): Random seed for reproducibility.
                Defaults to 42.
        """
        super().__init__()
        self.n_svd_components = n_svd_components
        self.random_state = random_state

    def fit(self, df: pd.DataFrame) -> Self:
        """Fit feature engineering by computing statistics and SVD factors.

        Fits a Truncated SVD model on the user-movie rating matrix and computes
        user and movie statistics. All fitted components are stored in
        self.encoders for use during transform.

        Args:
            df (pd.DataFrame): Input DataFrame containing user ratings and
                metadata. Must include columns: userId, movieId, rating,
                timestamp, and genres.

        Returns:
            Self: Returns self for method chaining.

        Raises:
            ValueError: If required columns are missing or SVD fails to fit.
        """
        # Fit SVD on the full dataset
        user_movie_matrix = df.pivot_table(
            values="rating", index="userId", columns="movieId", fill_value=0
        )

        svd = TruncatedSVD(
            n_components=self.n_svd_components, random_state=self.random_state
        )
        user_factors = svd.fit_transform(user_movie_matrix)
        movie_factors = svd.components_.T

        # Store encoders
        self.encoders["svd"] = svd
        self.encoders["user_factor_map"] = dict(
            zip(user_movie_matrix.index, user_factors)
        )
        self.encoders["movie_factor_map"] = dict(
            zip(user_movie_matrix.columns, movie_factors)
        )
        self.encoders["user_movie_matrix"] = user_movie_matrix

        # Compute and store statistics for transformation
        self._compute_statistics(df)

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """Transform DataFrame into feature matrix and target vector.

        Applies the fitted feature engineering transformations to new data.
        Must be called after fit().

        Args:
            df (pd.DataFrame): Input DataFrame to transform. Must contain
                the same columns as the DataFrame used in fit().

        Returns:
            Tuple[np.ndarray, pd.Series]: A tuple containing:
                - X (np.ndarray): Feature matrix with all engineered features.
                - y (pd.Series): Target values (rating column).

        Raises:
            KeyError: If transform is called before fit().
        """
        if not self.encoders:
            raise KeyError("Trying to transform without fit")
        X = self._engineer_features(df)
        y = df["rating"]
        return X.values, y

    def _compute_statistics(self, df: pd.DataFrame) -> None:
        """Compute and store user and movie statistics.

        Calculates and stores various statistics in self.encoders including:
            - User statistics: average rating, rating count, rating standard deviation
            - Movie statistics: average rating, rating count, rating standard deviation
            - Feature columns: list of all engineered feature names

        Args:
            df (pd.DataFrame): Input DataFrame containing user ratings.
                Must include userId, movieId, and rating columns.
        """
        # User statistics
        user_stats = (
            df.groupby("userId")["rating"].agg(["mean", "count", "std"]).fillna(0)
        )
        user_stats.columns = ["user_avg_rating", "user_rating_count", "user_rating_std"]
        self.encoders["user_stats"] = user_stats

        # Movie statistics
        movie_stats = (
            df.groupby("movieId")["rating"].agg(["mean", "count", "std"]).fillna(0)
        )
        movie_stats.columns = [
            "movie_avg_rating",
            "movie_rating_count",
            "movie_rating_std",
        ]
        self.encoders["movie_stats"] = movie_stats

        # Store feature columns
        df_with_features = self._engineer_features(df)
        self.encoders["feature_columns"] = df_with_features.columns.tolist()

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for the model.

        Applies feature engineering transformations to the input DataFrame
        using statistics and SVD factors stored in self.encoders from fit().
        Creates features including:
            - User statistics (average, count, standard deviation)
            - Movie statistics (average, count, standard deviation)
            - SVD latent features (collaborative filtering factors)
            - Genre one-hot encoding
            - Genre count
            - Rating deviation features (from user/movie averages)

        Args:
            df (pd.DataFrame): Input DataFrame to engineer features for.
                Must contain userId, movieId, rating, and genres columns.

        Returns:
            pd.DataFrame: DataFrame containing only the engineered feature
                columns, with all missing values filled with 0.
        """
        df = df.copy()

        # Feature 1: User statistics
        user_stats = self.encoders.get("user_stats")
        if user_stats is not None:
            df = df.merge(user_stats, on="userId", how="left")

        # Feature 2: Movie statistics
        movie_stats = self.encoders.get("movie_stats")
        if movie_stats is not None:
            df = df.merge(movie_stats, on="movieId", how="left")

        # Feature 3: SVD latent features (collaborative filtering)
        user_factor_map = self.encoders.get("user_factor_map")
        movie_factor_map = self.encoders.get("movie_factor_map")

        if user_factor_map is not None and movie_factor_map is not None:
            user_factors_array = np.array(
                [
                    user_factor_map.get(uid, np.zeros(self.n_svd_components))
                    for uid in df["userId"]
                ]
            )
            movie_factors_array = np.array(
                [
                    movie_factor_map.get(mid, np.zeros(self.n_svd_components))
                    for mid in df["movieId"]
                ]
            )

            for i in range(self.n_svd_components):
                df[f"user_factor_{i}"] = user_factors_array[:, i]
                df[f"movie_factor_{i}"] = movie_factors_array[:, i]

        # Feature 4: Genre one-hot encoding
        genres_dummies = df["genres"].str.get_dummies(sep="|")
        df = pd.concat([df, genres_dummies], axis=1)

        # Feature 5: Genre count
        df["num_genres"] = df["genres"].str.get_dummies(sep="|").sum(axis=1)

        # Feature 6: Rating deviation features
        if "user_avg_rating" in df.columns and "movie_avg_rating" in df.columns:
            df["user_rating_deviation"] = df["rating"] - df["user_avg_rating"]
            df["movie_rating_deviation"] = df["rating"] - df["movie_avg_rating"]

        # Select features (exclude non-feature columns)
        feature_columns = [
            col
            for col in df.columns
            if col not in ["userId", "movieId", "rating", "timestamp", "genres"]
        ]
        # Fill NaN values (for users/movies not seen during training)
        df[feature_columns] = df[feature_columns].fillna(0)

        return df[feature_columns]
