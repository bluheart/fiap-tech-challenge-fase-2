from ..base import FeatureEngineer
from typing import Self, Tuple
import numpy as np
import pandas as pd


class GradientBoostingFeatureEng(FeatureEngineer):
    """Feature engineering for Gradient Boosting model"""

    def __init__(self, random_state: int = 42):
        super().__init__()
        self.random_state = random_state

    def fit(self, df: pd.DataFrame) -> Self:
        """Fit feature engineering by computing and storing statistics"""

        # Compute and store statistics for transformation
        self._compute_statistics(df)

        # Store feature columns
        df_with_features = self._engineer_features(df)
        self.encoders["feature_columns"] = df_with_features.columns.tolist()

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """Transform DataFrame into feature matrix and target vector"""
        if not self.encoders:
            raise KeyError("Trying to transform without fit")

        X = self._engineer_features(df)
        y = df["rating"]
        return X.values, y

    def _compute_statistics(self, df: pd.DataFrame) -> None:
        """Compute and store all statistics needed for feature engineering"""

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        # User temporal patterns (day of week)
        # Ensure all 7 days are present by using reindex
        user_temporal = (
            df.groupby(["userId", df["timestamp"].dt.dayofweek])["rating"]
            .mean()
            .unstack(fill_value=0)
        )
        # Ensure all 7 days exist (0-6)
        for i in range(7):
            if i not in user_temporal.columns:
                user_temporal[i] = 0
        user_temporal = user_temporal.reindex(columns=range(7), fill_value=0)
        user_temporal.columns = [f"user_dow_{i}_avg" for i in range(7)]
        self.encoders["user_temporal"] = user_temporal

        # User temporal patterns (month)
        # Ensure all 12 months are present by using reindex
        user_monthly = (
            df.groupby(["userId", df["timestamp"].dt.month])["rating"]
            .mean()
            .unstack(fill_value=0)
        )
        # Ensure all 12 months exist (1-12)
        for i in range(1, 13):
            if i not in user_monthly.columns:
                user_monthly[i] = 0
        user_monthly = user_monthly.reindex(columns=range(1, 13), fill_value=0)
        user_monthly.columns = [f"user_month_{i}_avg" for i in range(1, 13)]
        self.encoders["user_monthly"] = user_monthly

        # Genre columns (from movies data)
        if "genres" in df.columns:
            genres_dummies = df["genres"].str.get_dummies(sep="|")
            self.encoders["genre_columns"] = genres_dummies.columns.tolist()

            # User genre preferences
            user_genre_prefs = {}
            for genre in genres_dummies.columns:
                df_temp = df.copy()
                df_temp[f"{genre}_rating"] = (
                    df_temp["genres"].str.contains(genre, regex=False).astype(int)
                ) * df_temp["rating"]
                user_genre_avg = df_temp.groupby("userId")[f"{genre}_rating"].sum() / (
                    df_temp.groupby("userId")["genres"].apply(
                        lambda x: x.str.contains(genre, regex=False).sum()
                    )
                    + 1e-6
                )
                user_genre_prefs[f"user_{genre}_preference"] = user_genre_avg
            self.encoders["user_genre_preferences"] = user_genre_prefs

        # Movie popularity
        movie_popularity = df.groupby("movieId").size()
        movie_popularity.name = "movie_popularity"
        self.encoders["movie_popularity"] = movie_popularity

        # User activity
        user_activity = df.groupby("userId").size()
        user_activity.name = "user_activity"
        self.encoders["user_activity"] = user_activity

        # Basic statistics
        self.encoders["user_avg_rating"] = df.groupby("userId")["rating"].mean()
        self.encoders["movie_avg_rating"] = df.groupby("movieId")["rating"].mean()
        self.encoders["user_rating_std"] = (
            df.groupby("userId")["rating"].std().fillna(0)
        )
        self.encoders["movie_rating_std"] = (
            df.groupby("movieId")["rating"].std().fillna(0)
        )

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for the model"""
        df = df.copy()

        # Feature 1: Time-based features
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["rating_month"] = df["timestamp"].dt.month
        df["rating_day"] = df["timestamp"].dt.day
        df["rating_hour"] = df["timestamp"].dt.hour
        df["rating_dayofweek"] = df["timestamp"].dt.dayofweek
        df["rating_quarter"] = df["timestamp"].dt.quarter
        df["rating_year"] = df["timestamp"].dt.year

        # Feature 2: Movie age (if movie_year exists)
        if "movie_year" in df.columns:
            df["movie_year"] = pd.to_numeric(df["movie_year"], errors="coerce")
            if "movie_age" not in df.columns:
                df["movie_age"] = 2024 - df["movie_year"]

        # Feature 3: User temporal patterns
        user_temporal = self.encoders.get("user_temporal")
        if user_temporal is not None:
            df = df.merge(user_temporal, on="userId", how="left")

        user_monthly = self.encoders.get("user_monthly")
        if user_monthly is not None:
            df = df.merge(user_monthly, on="userId", how="left")

        # Feature 4: Rolling statistics (user's recent ratings trend)
        df = df.sort_values(["userId", "timestamp"])
        df["user_rating_rolling_5"] = df.groupby("userId")["rating"].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df["user_rating_rolling_10"] = df.groupby("userId")["rating"].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )

        # Feature 5: Genre features (one-hot encoding)
        genre_columns = self.encoders.get("genre_columns", [])
        if "genres" in df.columns:
            genres_dummies = df["genres"].str.get_dummies(sep="|")
            # Ensure all genre columns from training are present
            for col in genre_columns:
                if col not in genres_dummies.columns:
                    genres_dummies[col] = 0
            genres_dummies = (
                genres_dummies[genre_columns] if genre_columns else genres_dummies
            )
            df = pd.concat([df, genres_dummies], axis=1)

        # Feature 6: User genre preferences
        user_genre_prefs = self.encoders.get("user_genre_preferences", {})
        for pref_name, pref_series in user_genre_prefs.items():
            df[pref_name] = df["userId"].map(pref_series).fillna(0)

        # Feature 7: Movie popularity
        movie_popularity = self.encoders.get("movie_popularity")
        if movie_popularity is not None:
            df["movie_popularity"] = df["movieId"].map(movie_popularity).fillna(0)

        # Feature 8: User activity
        user_activity = self.encoders.get("user_activity")
        if user_activity is not None:
            df["user_activity"] = df["userId"].map(user_activity).fillna(0)

        # Feature 9: Basic statistics
        user_avg = self.encoders.get("user_avg_rating")
        if user_avg is not None:
            df["user_avg_rating"] = df["userId"].map(user_avg)

        movie_avg = self.encoders.get("movie_avg_rating")
        if movie_avg is not None:
            df["movie_avg_rating"] = df["movieId"].map(movie_avg)

        user_std = self.encoders.get("user_rating_std")
        if user_std is not None:
            df["user_rating_std"] = df["userId"].map(user_std).fillna(0)

        movie_std = self.encoders.get("movie_rating_std")
        if movie_std is not None:
            df["movie_rating_std"] = df["movieId"].map(movie_std).fillna(0)

        # Select features (exclude non-feature columns)
        feature_columns = self.encoders.get("feature_columns", [])
        if feature_columns:
            # Ensure all feature columns exist
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            return df[feature_columns].fillna(0)
        else:
            # Fallback: exclude non-feature columns
            exclude_cols = [
                "userId",
                "movieId",
                "rating",
                "timestamp",
                "genres",
                "title",
                "movie_year",
            ]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            return df[feature_cols].fillna(0)
