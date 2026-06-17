from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from .base import FeatureEngineer
from typing import Self, Tuple
import numpy as np
import pandas as pd


class LinerRegressionFeatureEng(FeatureEngineer):
    def fit(self, df: pd.DataFrame) -> Self:
        X = self._engineer_features(df)
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(X)
        scaler = StandardScaler()
        scaler.fit_transform(X_poly)
        self.encoders["poly"] = poly
        self.encoders["scaler"] = scaler
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        X = self._engineer_features(df)
        y = df["rating"]
        poly = self.encoders["poly"]
        scaler = self.encoders["scaler"]
        X_poly = poly.transform(X)
        X_scaled = scaler.transform(X_poly)
        return X_scaled, y

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Feature 1: User statistics
        user_stats = (
            df.groupby("userId").agg({"rating": ["mean", "std", "count"]}).fillna(0)
        )
        user_stats.columns = ["user_avg_rating", "user_std_rating", "user_rating_count"]
        df = df.merge(user_stats, on="userId", how="left")

        # Feature 2: Movie statistics
        movie_stats = (
            df.groupby("movieId").agg({"rating": ["mean", "std", "count"]}).fillna(0)
        )
        movie_stats.columns = [
            "movie_avg_rating",
            "movie_std_rating",
            "movie_rating_count",
        ]
        df = df.merge(movie_stats, on="movieId", how="left")

        # Feature 3: Genre one-hot encoding
        genres_dummies = df["genres"].str.get_dummies(sep="|")
        df = pd.concat([df, genres_dummies], axis=1)

        # Feature 4: Time-based features
        df["rating_timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["rating_hour"] = df["rating_timestamp"].dt.hour
        df["rating_dayofweek"] = df["rating_timestamp"].dt.dayofweek

        # Feature 5: Interaction features
        df["user_movie_avg_interaction"] = (
            df["user_avg_rating"] * df["movie_avg_rating"]
        )

        # Select features (exclude non-feature columns)
        feature_columns = [
            col
            for col in df.columns
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
        return df[feature_columns]
