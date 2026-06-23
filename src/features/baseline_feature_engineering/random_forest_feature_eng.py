from sklearn.decomposition import TruncatedSVD
from ..base import FeatureEngineer
from typing import Self, Tuple
import numpy as np
import pandas as pd


class RandomForestFeatureEng(FeatureEngineer):
    def __init__(self, n_svd_components: int = 20, random_state: int = 42):
        super().__init__()
        self.n_svd_components = n_svd_components
        self.random_state = random_state

    def fit(self, df: pd.DataFrame) -> Self:
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
        if not self.encoders:
            raise KeyError("Trying to transform without fit")
        X = self._engineer_features(df)
        y = df["rating"]
        return X.values, y

    def _compute_statistics(self, df: pd.DataFrame) -> None:
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
