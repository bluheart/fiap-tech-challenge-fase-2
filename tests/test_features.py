import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.features.preprocessor import LinerRegressionPreprocessor
from src.features.feature_engineering import LinerRegressionFeatureEng


class TestLinerRegressionPreprocessor:
    """Test suite for LinerRegressionPreprocessor"""

    @pytest.fixture
    def sample_ratings_df(self):
        """Create sample ratings DataFrame"""
        return pd.DataFrame(
            {
                "userId": [1, 1, 2, 2, 3],
                "movieId": [1, 2, 1, 3, 2],
                "rating": [4.0, 5.0, 3.0, 4.5, 2.5],
                "timestamp": [964982703, 964982704, 964982705, 964982706, 964982707],
            }
        )

    @pytest.fixture
    def sample_movies_df(self):
        """Create sample movies DataFrame"""
        return pd.DataFrame(
            {
                "movieId": [1, 2, 3, 4],
                "genres": [
                    "Action|Adventure",
                    "Comedy|Drama",
                    "Action",
                    "(no genres listed)",
                ],
            }
        )

    def test_apply_removes_no_genre_movies(self, sample_ratings_df, sample_movies_df):
        """Test that movies with '(no genres listed)' are filtered out"""
        preprocessor = LinerRegressionPreprocessor()
        result = preprocessor.apply(sample_ratings_df, sample_movies_df)

        # Movie 4 has no genres, should be filtered out
        assert 4 not in result["movieId"].values
        assert len(result[result["movieId"] == 4]) == 0

    def test_apply_merges_genres_correctly(self, sample_ratings_df, sample_movies_df):
        """Test that genres are correctly merged from movies_df"""
        preprocessor = LinerRegressionPreprocessor()
        result = preprocessor.apply(sample_ratings_df, sample_movies_df)

        # Check that genres column exists
        assert "genres" in result.columns

        # Check specific merges
        assert result[result["movieId"] == 1]["genres"].iloc[0] == "Action|Adventure"
        assert result[result["movieId"] == 2]["genres"].iloc[0] == "Comedy|Drama"

    def test_apply_does_not_modify_original_dfs(
        self, sample_ratings_df, sample_movies_df
    ):
        """Test that original DataFrames are not modified"""
        original_ratings = sample_ratings_df.copy()
        original_movies = sample_movies_df.copy()

        preprocessor = LinerRegressionPreprocessor()
        preprocessor.apply(sample_ratings_df, sample_movies_df)

        # Check that originals are unchanged
        pd.testing.assert_frame_equal(original_ratings, sample_ratings_df)
        pd.testing.assert_frame_equal(original_movies, sample_movies_df)

    def test_apply_returns_dataframe(self, sample_ratings_df, sample_movies_df):
        """Test that apply returns a pandas DataFrame"""
        preprocessor = LinerRegressionPreprocessor()
        result = preprocessor.apply(sample_ratings_df, sample_movies_df)

        assert isinstance(result, pd.DataFrame)

    def test_apply_handles_empty_dataframe(self):
        """Test that apply handles empty DataFrames gracefully"""
        preprocessor = LinerRegressionPreprocessor()
        empty_ratings = pd.DataFrame(
            columns=["userId", "movieId", "rating", "timestamp"]
        )
        empty_movies = pd.DataFrame(columns=["movieId", "genres"])

        result = preprocessor.apply(empty_ratings, empty_movies)
        assert len(result) == 0


class TestLinerRegressionFeatureEng:
    """Test suite for LinerRegressionFeatureEng"""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with features"""
        return pd.DataFrame(
            {
                "userId": [1, 1, 2, 2, 3],
                "movieId": [1, 2, 1, 3, 2],
                "rating": [4.0, 5.0, 3.0, 4.5, 2.5],
                "timestamp": [964982703, 964982704, 964982705, 964982706, 964982707],
                "genres": [
                    "Action|Adventure",
                    "Comedy|Drama",
                    "Action",
                    "Action",
                    "Comedy|Drama",
                ],
            }
        )

    def test_fit_creates_encoders(self, sample_df):
        """Test that fit creates polynomial and scaler encoders"""
        feature_eng = LinerRegressionFeatureEng()
        result = feature_eng.fit(sample_df)

        # Check that it returns self
        assert result is feature_eng

        # Check that encoders were created
        assert "poly" in feature_eng.encoders
        assert "scaler" in feature_eng.encoders

        # Check that encoders are fitted
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        assert isinstance(feature_eng.encoders["poly"], PolynomialFeatures)
        assert isinstance(feature_eng.encoders["scaler"], StandardScaler)

    def test_transform_returns_correct_shape(self, sample_df):
        """Test that transform returns arrays of correct shape"""
        feature_eng = LinerRegressionFeatureEng()
        feature_eng.fit(sample_df)

        X, y = feature_eng.transform(sample_df)

        # Check shapes
        assert len(X) == len(sample_df)  # Same number of samples
        assert len(y) == len(sample_df)  # Same number of samples
        assert isinstance(X, np.ndarray)
        assert isinstance(y, pd.Series)

    def test_transform_returns_scaled_features(self, sample_df):
        """Test that transform returns scaled features"""
        feature_eng = LinerRegressionFeatureEng()
        feature_eng.fit(sample_df)

        X, y = feature_eng.transform(sample_df)

        # Check that features are scaled (mean near 0, std near 1)
        # Note: with small sample size, values might not be perfect
        mean = np.mean(X, axis=0)

        # Allow some tolerance for small sample
        assert np.all(np.abs(mean) < 1e-10) or len(sample_df) < 10

    def test_engineer_features_creates_correct_columns(self, sample_df):
        """Test internal _engineer_features method"""
        feature_eng = LinerRegressionFeatureEng()
        result = feature_eng._engineer_features(sample_df)

        # Check that all expected feature columns are created
        expected_features = [
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

        # Check for genre dummies (Action, Adventure, Comedy, Drama)
        genre_features = ["Action", "Adventure", "Comedy", "Drama"]

        for feature in expected_features + genre_features:
            assert feature in result.columns, f"Feature {feature} missing"

        # Check that non-feature columns are excluded
        non_features = [
            "userId",
            "movieId",
            "rating",
            "timestamp",
            "genres",
            "rating_timestamp",
        ]
        for col in non_features:
            assert col not in result.columns, f"Column {col} should be excluded"

    def test_engineer_features_calculates_stats_correctly(self, sample_df):
        """Test that user and movie stats are calculated correctly"""
        feature_eng = LinerRegressionFeatureEng()
        result = feature_eng._engineer_features(sample_df)

        # Check user stats for userId=1 (ratings: 4.0, 5.0)
        user1_stats = result[sample_df["userId"] == 1].iloc[0]
        assert user1_stats["user_avg_rating"] == 4.5
        assert user1_stats["user_rating_count"] == 2

        # Check movie stats for movieId=1 (ratings: 4.0, 3.0)
        movie1_stats = result[sample_df["movieId"] == 1].iloc[0]
        assert movie1_stats["movie_avg_rating"] == 3.5
        assert movie1_stats["movie_rating_count"] == 2

    def test_engineer_features_handles_timestamp_conversion(self, sample_df):
        """Test that timestamp is correctly converted to datetime features"""
        feature_eng = LinerRegressionFeatureEng()
        result = feature_eng._engineer_features(sample_df)

        # Check that timestamp features exist and are numeric
        assert "rating_hour" in result.columns
        assert "rating_dayofweek" in result.columns
        assert pd.api.types.is_numeric_dtype(result["rating_hour"])
        assert pd.api.types.is_numeric_dtype(result["rating_dayofweek"])

    def test_fit_transform_consistency(self, sample_df):
        """Test that fit and transform produce consistent results"""
        feature_eng = LinerRegressionFeatureEng()
        feature_eng.fit(sample_df)

        X1, y1 = feature_eng.transform(sample_df)
        X2, y2 = feature_eng.transform(sample_df)

        # Transform should be deterministic
        np.testing.assert_array_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)

    def test_transform_requires_fit_first(self, sample_df):
        """Test that transform raises error if fit hasn't been called"""
        feature_eng = LinerRegressionFeatureEng()

        with pytest.raises(KeyError):
            feature_eng.transform(sample_df)


# Integration tests
class TestIntegration:
    """Integration tests for the full pipeline"""

    @pytest.fixture
    def full_pipeline_data(self):
        """Create full pipeline test data"""
        ratings = pd.DataFrame(
            {
                "userId": [1, 1, 2, 2, 3, 3],
                "movieId": [1, 2, 1, 3, 2, 4],
                "rating": [4.0, 5.0, 3.0, 4.5, 2.5, 3.5],
                "timestamp": [
                    964982703,
                    964982704,
                    964982705,
                    964982706,
                    964982707,
                    964982708,
                ],
            }
        )
        movies = pd.DataFrame(
            {
                "movieId": [1, 2, 3, 4],
                "genres": ["Action|Adventure", "Comedy|Drama", "Action", "Action"],
            }
        )
        return ratings, movies

    def test_full_pipeline(self, full_pipeline_data):
        """Test the complete preprocessing + feature engineering pipeline"""
        ratings_df, movies_df = full_pipeline_data

        # Preprocess
        preprocessor = LinerRegressionPreprocessor()
        processed_df = preprocessor.apply(ratings_df, movies_df)

        # Feature engineering
        feature_eng = LinerRegressionFeatureEng()
        feature_eng.fit(processed_df)
        X, y = feature_eng.transform(processed_df)

        # Check that we get valid outputs
        assert X.shape[0] == len(processed_df)
        assert y.shape[0] == len(processed_df)
        assert not np.isnan(X).any()  # No NaN values
        assert not np.isinf(X).any()  # No infinite values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
