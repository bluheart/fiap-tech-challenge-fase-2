from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from ..base import FeatureEngineer
from typing import Tuple, Self, Dict, Any
import numpy as np
import pandas as pd


class LinearRegressionFeatureEng(FeatureEngineer):
    """Feature engineering class for Linear Regression model.

    This class handles feature engineering including polynomial features,
    scaling, and various derived features from movie rating data.

    Attributes:
        encoders: Dictionary to store fitted transformers.
    """

    def __init__(self) -> None:
        """Initialize the feature engineer with empty encoders."""
        self.encoders: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> Self:
        """Fit feature engineering transformers on the training data.

        Args:
            df: Input DataFrame containing user-movie rating data with columns:
                userId, movieId, rating, timestamp, genres.

        Returns:
            Self, allowing for method chaining.
        """
        X = self._engineer_features(df)

        # Create polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(X)

        # Fit scaler on polynomial features
        scaler = StandardScaler()
        scaler.fit(X_poly)  # Changed from fit_transform to fit

        # Store fitted transformers
        self.encoders["poly"] = poly
        self.encoders["scaler"] = scaler
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """Transform data using fitted feature engineering transformers.

        Args:
            df: Input DataFrame containing user-movie rating data.

        Returns:
            Tuple containing:
                - X_scaled: Transformed feature matrix as numpy array.
                - y: Target variable (ratings) as pandas Series.

        Raises:
            KeyError: If fit() hasn't been called before transform().
        """
        if not self.encoders:
            raise KeyError("Trying to transform without fit")

        X = self._engineer_features(df)
        y = df["rating"]

        poly = self.encoders["poly"]
        scaler = self.encoders["scaler"]

        X_poly = poly.transform(X)
        X_scaled = scaler.transform(X_poly)

        return X_scaled, y

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw movie rating data.

        Args:
            df: Input DataFrame with columns: userId, movieId, rating,
                timestamp, genres.

        Returns:
            DataFrame with engineered features only (excluding raw columns).
        """
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
