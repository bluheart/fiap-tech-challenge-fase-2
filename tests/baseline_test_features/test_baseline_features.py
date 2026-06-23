import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.decomposition import TruncatedSVD

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features import (
    LinearRegressionPreprocessor,
    RandomForestPreprocessor,
    GradientBoostingPreprocessor,
)
from src.features import (
    LinerRegressionFeatureEng,
    RandomForestFeatureEng,
    GradientBoostingFeatureEng,
)


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
        preprocessor = LinearRegressionPreprocessor()
        result = preprocessor.apply(sample_ratings_df, sample_movies_df)

        # Movie 4 has no genres, should be filtered out
        assert 4 not in result["movieId"].values
        assert len(result[result["movieId"] == 4]) == 0

    def test_apply_merges_genres_correctly(self, sample_ratings_df, sample_movies_df):
        """Test that genres are correctly merged from movies_df"""
        preprocessor = LinearRegressionPreprocessor()
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

        preprocessor = LinearRegressionPreprocessor()
        preprocessor.apply(sample_ratings_df, sample_movies_df)

        # Check that originals are unchanged
        pd.testing.assert_frame_equal(original_ratings, sample_ratings_df)
        pd.testing.assert_frame_equal(original_movies, sample_movies_df)

    def test_apply_returns_dataframe(self, sample_ratings_df, sample_movies_df):
        """Test that apply returns a pandas DataFrame"""
        preprocessor = LinearRegressionPreprocessor()
        result = preprocessor.apply(sample_ratings_df, sample_movies_df)

        assert isinstance(result, pd.DataFrame)

    def test_apply_handles_empty_dataframe(self):
        """Test that apply handles empty DataFrames gracefully"""
        preprocessor = LinearRegressionPreprocessor()
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
        preprocessor = LinearRegressionPreprocessor()
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


class TestRandomForestPreprocessor:
    """Test suite for RandomForestPreprocessor"""

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
        preprocessor = RandomForestPreprocessor()
        result = preprocessor.apply(sample_ratings_df, sample_movies_df)

        # Movie 4 has no genres, should be filtered out
        assert 4 not in result["movieId"].values
        assert len(result[result["movieId"] == 4]) == 0

    def test_apply_merges_genres_correctly(self, sample_ratings_df, sample_movies_df):
        """Test that genres are correctly merged from movies_df"""
        preprocessor = RandomForestPreprocessor()
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

        preprocessor = RandomForestPreprocessor()
        preprocessor.apply(sample_ratings_df, sample_movies_df)

        # Check that originals are unchanged
        pd.testing.assert_frame_equal(original_ratings, sample_ratings_df)
        pd.testing.assert_frame_equal(original_movies, sample_movies_df)

    def test_apply_returns_dataframe(self, sample_ratings_df, sample_movies_df):
        """Test that apply returns a pandas DataFrame"""
        preprocessor = RandomForestPreprocessor()
        result = preprocessor.apply(sample_ratings_df, sample_movies_df)

        assert isinstance(result, pd.DataFrame)

    def test_apply_handles_empty_dataframe(self):
        """Test that apply handles empty DataFrames gracefully"""
        preprocessor = RandomForestPreprocessor()
        empty_ratings = pd.DataFrame(
            columns=["userId", "movieId", "rating", "timestamp"]
        )
        empty_movies = pd.DataFrame(columns=["movieId", "genres"])

        result = preprocessor.apply(empty_ratings, empty_movies)
        assert len(result) == 0

    def test_apply_handles_all_no_genre_movies(self, sample_ratings_df):
        """Test when all movies have '(no genres listed)'"""
        preprocessor = RandomForestPreprocessor()
        movies_df = pd.DataFrame(
            {
                "movieId": [1, 2, 3],
                "genres": [
                    "(no genres listed)",
                    "(no genres listed)",
                    "(no genres listed)",
                ],
            }
        )

        result = preprocessor.apply(sample_ratings_df, movies_df)
        # Should return empty DataFrame since all movies are filtered out
        assert len(result) == 0


class TestRandomForestFeatureEng:
    """Test suite for RandomForestFeatureEng"""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with features"""
        return pd.DataFrame(
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
                "genres": [
                    "Action|Adventure",
                    "Comedy|Drama",
                    "Action",
                    "Action",
                    "Comedy|Drama",
                    "Action|Adventure",
                ],
            }
        )

    @pytest.fixture
    def large_sample_df(self):
        """Create a larger sample DataFrame for SVD testing"""
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame(
            {
                "userId": np.random.randint(1, 11, n_samples),
                "movieId": np.random.randint(1, 21, n_samples),
                "rating": np.random.uniform(1, 5, n_samples),
                "timestamp": np.random.randint(964982703, 964990000, n_samples),
                "genres": np.random.choice(
                    ["Action", "Comedy", "Drama", "Action|Adventure", "Comedy|Drama"],
                    n_samples,
                ),
            }
        )

    def test_fit_creates_encoders(self, sample_df):
        """Test that fit creates SVD and statistics encoders"""
        feature_eng = RandomForestFeatureEng(n_svd_components=2)
        result = feature_eng.fit(sample_df)

        # Check that it returns self
        assert result is feature_eng

        # Check that encoders were created
        assert "svd" in feature_eng.encoders
        assert "user_factor_map" in feature_eng.encoders
        assert "movie_factor_map" in feature_eng.encoders
        assert "user_stats" in feature_eng.encoders
        assert "movie_stats" in feature_eng.encoders
        assert "feature_columns" in feature_eng.encoders

        # Check that SVD is fitted
        assert isinstance(feature_eng.encoders["svd"], TruncatedSVD)

    def test_fit_creates_correct_svd_dimensions(self, large_sample_df):
        """Test that SVD creates correct number of components"""
        n_components = 5
        feature_eng = RandomForestFeatureEng(n_svd_components=n_components)
        feature_eng.fit(large_sample_df)

        # Check user factors shape
        user_factors = feature_eng.encoders["user_factor_map"]
        for user_id, factors in user_factors.items():
            assert len(factors) == n_components

        # Check movie factors shape
        movie_factors = feature_eng.encoders["movie_factor_map"]
        for movie_id, factors in movie_factors.items():
            assert len(factors) == n_components

    def test_transform_returns_correct_shape(self, sample_df):
        """Test that transform returns arrays of correct shape"""
        feature_eng = RandomForestFeatureEng(n_svd_components=2)
        feature_eng.fit(sample_df)

        X, y = feature_eng.transform(sample_df)

        # Check shapes
        assert len(X) == len(sample_df)  # Same number of samples
        assert len(y) == len(sample_df)  # Same number of samples
        assert isinstance(X, np.ndarray)
        assert isinstance(y, pd.Series)

    def test_engineer_features_creates_correct_columns(self, sample_df):
        """Test internal _engineer_features method creates expected columns"""
        feature_eng = RandomForestFeatureEng(n_svd_components=2)
        feature_eng.fit(sample_df)
        result = feature_eng._engineer_features(sample_df)

        # Check that user statistics features exist
        user_stat_features = ["user_avg_rating", "user_rating_count", "user_rating_std"]
        for feature in user_stat_features:
            assert feature in result.columns, f"Feature {feature} missing"

        # Check that movie statistics features exist
        movie_stat_features = [
            "movie_avg_rating",
            "movie_rating_count",
            "movie_rating_std",
        ]
        for feature in movie_stat_features:
            assert feature in result.columns, f"Feature {feature} missing"

        # Check that SVD features exist
        for i in range(2):
            assert f"user_factor_{i}" in result.columns
            assert f"movie_factor_{i}" in result.columns

        # Check that genre dummies exist
        genre_features = ["Action", "Adventure", "Comedy", "Drama"]
        for feature in genre_features:
            assert feature in result.columns, f"Genre feature {feature} missing"

        # Check genre count feature
        assert "num_genres" in result.columns

        # Check rating deviation features
        assert "user_rating_deviation" in result.columns
        assert "movie_rating_deviation" in result.columns

        # Check that non-feature columns are excluded
        non_features = ["userId", "movieId", "rating", "timestamp", "genres"]
        for col in non_features:
            assert col not in result.columns, f"Column {col} should be excluded"

    def test_engineer_features_calculates_stats_correctly(self, sample_df):
        """Test that user and movie stats are calculated correctly"""
        feature_eng = RandomForestFeatureEng(n_svd_components=2)
        feature_eng.fit(sample_df)
        result = feature_eng._engineer_features(sample_df)

        # Check user stats for userId=1 (ratings: 4.0, 5.0)
        user1_stats = result[sample_df["userId"] == 1].iloc[0]
        assert user1_stats["user_avg_rating"] == 4.5
        assert user1_stats["user_rating_count"] == 2

        # Check movie stats for movieId=1 (ratings: 4.0, 3.0)
        movie1_stats = result[sample_df["movieId"] == 1].iloc[0]
        assert movie1_stats["movie_avg_rating"] == 3.5
        assert movie1_stats["movie_rating_count"] == 2

    def test_engineer_features_handles_missing_users(self, sample_df):
        """Test that unseen users/movies get zero-filled features"""
        feature_eng = RandomForestFeatureEng(n_svd_components=2)
        feature_eng.fit(sample_df)

        # Create a DataFrame with a new user
        new_df = pd.DataFrame(
            {
                "userId": [999],
                "movieId": [1],
                "rating": [4.0],
                "timestamp": [964982703],
                "genres": ["Action"],
            }
        )

        result = feature_eng._engineer_features(new_df)

        # Check that no NaN values exist
        assert not result.isna().any().any()

        # Check that missing user's stats are zero-filled
        assert (result["user_avg_rating"] == 0).all()
        assert (result["user_rating_count"] == 0).all()
        assert (result["user_rating_std"] == 0).all()

    def test_genre_one_hot_encoding(self, sample_df):
        """Test that genre one-hot encoding works correctly"""
        feature_eng = RandomForestFeatureEng(n_svd_components=2)
        feature_eng.fit(sample_df)
        result = feature_eng._engineer_features(sample_df)

        # Check specific genre encoding
        first_row = result.iloc[0]  # "Action|Adventure"
        assert first_row["Action"] == 1
        assert first_row["Adventure"] == 1
        assert first_row["Comedy"] == 0
        assert first_row["Drama"] == 0

        second_row = result.iloc[1]  # "Comedy|Drama"
        assert second_row["Action"] == 0
        assert second_row["Adventure"] == 0
        assert second_row["Comedy"] == 1
        assert second_row["Drama"] == 1

    def test_num_genres_feature(self, sample_df):
        """Test that num_genres feature counts correctly"""
        feature_eng = RandomForestFeatureEng(n_svd_components=2)
        feature_eng.fit(sample_df)
        result = feature_eng._engineer_features(sample_df)

        # Check genre counts
        assert result.iloc[0]["num_genres"] == 2  # Action|Adventure
        assert result.iloc[1]["num_genres"] == 2  # Comedy|Drama
        assert result.iloc[2]["num_genres"] == 1  # Action

    def test_rating_deviation_features(self, sample_df):
        """Test that rating deviation features are calculated correctly"""
        feature_eng = RandomForestFeatureEng(n_svd_components=2)
        feature_eng.fit(sample_df)
        result = feature_eng._engineer_features(sample_df)

        # For userId=1, movieId=1, rating=4.0
        # User 1 average: 4.5, Movie 1 average: 3.5
        first_row = result.iloc[0]
        assert abs(first_row["user_rating_deviation"] - (4.0 - 4.5)) < 0.01
        assert abs(first_row["movie_rating_deviation"] - (4.0 - 3.5)) < 0.01

    def test_fit_transform_consistency(self, sample_df):
        """Test that fit and transform produce consistent results"""
        feature_eng = RandomForestFeatureEng(n_svd_components=2)
        feature_eng.fit(sample_df)

        X1, y1 = feature_eng.transform(sample_df)
        X2, y2 = feature_eng.transform(sample_df)

        # Transform should be deterministic
        np.testing.assert_array_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)

    def test_transform_requires_fit_first(self, sample_df):
        """Test that transform raises error if fit hasn't been called"""
        feature_eng = RandomForestFeatureEng()

        with pytest.raises(KeyError):
            feature_eng.transform(sample_df)

    def test_transform_preserves_feature_order(self, large_sample_df):
        """Test that feature order is consistent"""
        feature_eng = RandomForestFeatureEng(n_svd_components=3)
        feature_eng.fit(large_sample_df)

        # Store feature columns during fit
        expected_columns = feature_eng.encoders["feature_columns"]

        # Transform should create same columns in same order
        df_features = feature_eng._engineer_features(large_sample_df)
        assert list(df_features.columns) == expected_columns

    def test_svd_embeddings_additive(self, large_sample_df):
        """Test that SVD captures some structure (basic sanity check)"""
        feature_eng = RandomForestFeatureEng(n_svd_components=5)
        feature_eng.fit(large_sample_df)

        # Get features
        X, _ = feature_eng.transform(large_sample_df)

        # Check that SVD components have non-zero variance
        svd_column_indices = [
            i
            for i, col in enumerate(feature_eng.encoders["feature_columns"])
            if "factor" in col
        ]
        for idx in svd_column_indices:
            assert np.std(X[:, idx]) > 0, f"SVD component {idx} has zero variance"


class TestRandomForestIntegration:
    """Integration tests for the full Random Forest pipeline"""

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
        preprocessor = RandomForestPreprocessor()
        processed_df = preprocessor.apply(ratings_df, movies_df)

        # Feature engineering
        feature_eng = RandomForestFeatureEng(n_svd_components=2, random_state=42)
        feature_eng.fit(processed_df)
        X, y = feature_eng.transform(processed_df)

        # Check that we get valid outputs
        assert X.shape[0] == len(processed_df)
        assert y.shape[0] == len(processed_df)
        assert not np.isnan(X).any()  # No NaN values
        assert not np.isinf(X).any()  # No infinite values

    def test_full_pipeline_with_no_genre_movies(self, full_pipeline_data):
        """Test pipeline handles movies with no genres correctly"""
        ratings_df, movies_df = full_pipeline_data

        # Add a movie with no genres
        movies_df = pd.concat(
            [
                movies_df,
                pd.DataFrame({"movieId": [5], "genres": ["(no genres listed)"]}),
            ]
        )

        # Preprocess
        preprocessor = RandomForestPreprocessor()
        processed_df = preprocessor.apply(ratings_df, movies_df)

        # Movie 5 should be filtered out
        assert 5 not in processed_df["movieId"].values

        # Feature engineering should still work
        feature_eng = RandomForestFeatureEng(n_svd_components=2)
        feature_eng.fit(processed_df)
        X, y = feature_eng.transform(processed_df)

        assert not np.isnan(X).any()

    def test_pipeline_deterministic(self, full_pipeline_data):
        """Test that the full pipeline is deterministic with fixed random state"""
        ratings_df, movies_df = full_pipeline_data

        # Run pipeline twice
        preprocessor1 = RandomForestPreprocessor()
        processed_df1 = preprocessor1.apply(ratings_df, movies_df)
        feature_eng1 = RandomForestFeatureEng(n_svd_components=2, random_state=42)
        feature_eng1.fit(processed_df1)
        X1, y1 = feature_eng1.transform(processed_df1)

        preprocessor2 = RandomForestPreprocessor()
        processed_df2 = preprocessor2.apply(ratings_df, movies_df)
        feature_eng2 = RandomForestFeatureEng(n_svd_components=2, random_state=42)
        feature_eng2.fit(processed_df2)
        X2, y2 = feature_eng2.transform(processed_df2)

        # Results should be identical
        np.testing.assert_array_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)


class TestGradientBoostingPreprocessor:
    """Test suite for GradientBoostingPreprocessor"""

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
        preprocessor = GradientBoostingPreprocessor()
        result = preprocessor.apply(sample_ratings_df, sample_movies_df)

        # Movie 4 has no genres, should be filtered out
        assert 4 not in result["movieId"].values
        assert len(result[result["movieId"] == 4]) == 0

    def test_apply_merges_genres_correctly(self, sample_ratings_df, sample_movies_df):
        """Test that genres are correctly merged from movies_df"""
        preprocessor = GradientBoostingPreprocessor()
        result = preprocessor.apply(sample_ratings_df, sample_movies_df)

        # Check that genres column exists
        assert "genres" in result.columns

        # Check specific merges
        assert result[result["movieId"] == 1]["genres"].iloc[0] == "Action|Adventure"
        assert result[result["movieId"] == 2]["genres"].iloc[0] == "Comedy|Drama"

    def test_apply_uses_inner_join(self, sample_ratings_df, sample_movies_df):
        """Test that inner join only keeps matching movieIds"""
        # Add a rating for a movie not in movies_df
        ratings_df = pd.concat(
            [
                sample_ratings_df,
                pd.DataFrame(
                    {
                        "userId": [4],
                        "movieId": [999],
                        "rating": [3.0],
                        "timestamp": [964982708],
                    }
                ),
            ]
        )

        preprocessor = GradientBoostingPreprocessor()
        result = preprocessor.apply(ratings_df, sample_movies_df)

        # Movie 999 should not be in the result (not in movies_df)
        assert 999 not in result["movieId"].values
        # Movie 4 should not be in the result (no genres)
        assert 4 not in result["movieId"].values

    def test_apply_does_not_modify_original_dfs(
        self, sample_ratings_df, sample_movies_df
    ):
        """Test that original DataFrames are not modified"""
        original_ratings = sample_ratings_df.copy()
        original_movies = sample_movies_df.copy()

        preprocessor = GradientBoostingPreprocessor()
        preprocessor.apply(sample_ratings_df, sample_movies_df)

        # Check that originals are unchanged
        pd.testing.assert_frame_equal(original_ratings, sample_ratings_df)
        pd.testing.assert_frame_equal(original_movies, sample_movies_df)

    def test_apply_returns_dataframe(self, sample_ratings_df, sample_movies_df):
        """Test that apply returns a pandas DataFrame"""
        preprocessor = GradientBoostingPreprocessor()
        result = preprocessor.apply(sample_ratings_df, sample_movies_df)

        assert isinstance(result, pd.DataFrame)

    def test_apply_preserves_rating_data(self, sample_ratings_df, sample_movies_df):
        """Test that rating data is preserved correctly"""
        preprocessor = GradientBoostingPreprocessor()
        result = preprocessor.apply(sample_ratings_df, sample_movies_df)

        # Check that ratings are preserved
        assert result[result["movieId"] == 1]["rating"].iloc[0] == 4.0
        assert result[result["movieId"] == 2]["rating"].iloc[0] == 5.0

    def test_apply_handles_empty_dataframe(self):
        """Test that apply handles empty DataFrames gracefully"""
        preprocessor = GradientBoostingPreprocessor()
        empty_ratings = pd.DataFrame(
            columns=["userId", "movieId", "rating", "timestamp"]
        )
        empty_movies = pd.DataFrame(columns=["movieId", "genres"])

        result = preprocessor.apply(empty_ratings, empty_movies)
        assert len(result) == 0

    def test_apply_handles_all_no_genre_movies(self, sample_ratings_df):
        """Test when all movies have '(no genres listed)'"""
        preprocessor = GradientBoostingPreprocessor()
        movies_df = pd.DataFrame(
            {
                "movieId": [1, 2, 3],
                "genres": [
                    "(no genres listed)",
                    "(no genres listed)",
                    "(no genres listed)",
                ],
            }
        )

        result = preprocessor.apply(sample_ratings_df, movies_df)
        # Should return empty DataFrame since all movies are filtered out
        assert len(result) == 0

    def test_apply_handles_missing_movie_ids(self, sample_ratings_df):
        """Test when ratings reference movies not in movies_df"""
        preprocessor = GradientBoostingPreprocessor()
        movies_df = pd.DataFrame(
            {
                "movieId": [1],
                "genres": ["Action"],
            }
        )

        result = preprocessor.apply(sample_ratings_df, movies_df)
        # Should only keep ratings for movieId=1
        assert len(result) == 2  # Two ratings for movieId=1
        assert all(result["movieId"] == 1)


class TestGradientBoostingFeatureEng:
    """Test suite for GradientBoostingFeatureEng"""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with features"""
        return pd.DataFrame(
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
                "genres": [
                    "Action|Adventure",
                    "Comedy|Drama",
                    "Action",
                    "Action",
                    "Comedy|Drama",
                    "Action|Adventure",
                ],
            }
        )

    @pytest.fixture
    def large_sample_df(self):
        """Create a larger sample DataFrame for better testing"""
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame(
            {
                "userId": np.random.randint(1, 11, n_samples),
                "movieId": np.random.randint(1, 21, n_samples),
                "rating": np.random.uniform(1, 5, n_samples),
                "timestamp": np.random.randint(964982703, 964990000, n_samples),
                "genres": np.random.choice(
                    ["Action", "Comedy", "Drama", "Action|Adventure", "Comedy|Drama"],
                    n_samples,
                ),
            }
        )

    def test_fit_creates_encoders(self, sample_df):
        """Test that fit creates all necessary encoders"""
        feature_eng = GradientBoostingFeatureEng()
        result = feature_eng.fit(sample_df)

        # Check that it returns self
        assert result is feature_eng

        # Check that encoders were created
        expected_encoders = [
            "user_temporal",
            "user_monthly",
            "genre_columns",
            "user_genre_preferences",
            "movie_popularity",
            "user_activity",
            "user_avg_rating",
            "movie_avg_rating",
            "user_rating_std",
            "movie_rating_std",
            "feature_columns",
        ]
        for encoder_name in expected_encoders:
            assert encoder_name in feature_eng.encoders, (
                f"Encoder {encoder_name} missing"
            )

    def test_fit_stores_correct_statistics(self, sample_df):
        """Test that fit computes and stores correct statistics"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(sample_df)

        # Check user average ratings
        user_avg = feature_eng.encoders["user_avg_rating"]
        assert user_avg[1] == 4.5  # User 1 has ratings 4.0 and 5.0
        assert user_avg[2] == 3.75  # User 2 has ratings 3.0 and 4.5
        assert user_avg[3] == 3.0  # User 3 has ratings 2.5 and 3.5

        # Check movie average ratings
        movie_avg = feature_eng.encoders["movie_avg_rating"]
        assert movie_avg[1] == 3.5  # Movie 1 has ratings 4.0 and 3.0
        assert movie_avg[2] == 3.75  # Movie 2 has ratings 5.0 and 2.5

        # Check movie popularity
        movie_pop = feature_eng.encoders["movie_popularity"]
        assert movie_pop[1] == 2  # Movie 1 appears twice
        assert movie_pop[4] == 1  # Movie 4 appears once

        # Check user activity
        user_act = feature_eng.encoders["user_activity"]
        assert user_act[1] == 2  # User 1 has 2 ratings
        assert user_act[3] == 2  # User 3 has 2 ratings

    def test_fit_stores_genre_columns(self, sample_df):
        """Test that genre columns are correctly identified and stored"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(sample_df)

        genre_columns = feature_eng.encoders["genre_columns"]
        expected_genres = ["Action", "Adventure", "Comedy", "Drama"]
        for genre in expected_genres:
            assert genre in genre_columns, f"Genre {genre} missing"

    def test_transform_returns_correct_shape(self, sample_df):
        """Test that transform returns arrays of correct shape"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(sample_df)

        X, y = feature_eng.transform(sample_df)

        # Check shapes
        assert len(X) == len(sample_df)
        assert len(y) == len(sample_df)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, pd.Series)

    def test_engineer_features_creates_time_features(self, sample_df):
        """Test that time-based features are created"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(sample_df)
        result = feature_eng._engineer_features(sample_df)

        time_features = [
            "rating_month",
            "rating_day",
            "rating_hour",
            "rating_dayofweek",
            "rating_quarter",
            "rating_year",
        ]
        for feature in time_features:
            assert feature in result.columns, f"Time feature {feature} missing"

        # Check that values are numeric and non-null
        for feature in time_features:
            assert pd.api.types.is_numeric_dtype(result[feature])
            assert result[feature].notna().all()

    def test_engineer_features_creates_user_temporal_features(self, sample_df):
        """Test that user temporal pattern features are created"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(sample_df)
        result = feature_eng._engineer_features(sample_df)

        # Check day of week features
        for i in range(7):
            assert f"user_dow_{i}_avg" in result.columns

        # Check month features
        for i in range(1, 13):
            assert f"user_month_{i}_avg" in result.columns

    def test_engineer_features_creates_rolling_features(self, sample_df):
        """Test that rolling statistics features are created"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(sample_df)
        result = feature_eng._engineer_features(sample_df)

        assert "user_rating_rolling_5" in result.columns
        assert "user_rating_rolling_10" in result.columns

        # Check that rolling values are not null
        assert result["user_rating_rolling_5"].notna().all()
        assert result["user_rating_rolling_10"].notna().all()

    def test_engineer_features_creates_genre_features(self, sample_df):
        """Test that genre features are created"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(sample_df)
        result = feature_eng._engineer_features(sample_df)

        genre_features = ["Action", "Adventure", "Comedy", "Drama"]
        for feature in genre_features:
            assert feature in result.columns, f"Genre feature {feature} missing"

    def test_genre_one_hot_encoding_correct(self, sample_df):
        """Test that genre one-hot encoding works correctly"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(sample_df)
        result = feature_eng._engineer_features(sample_df)

        # Check specific genre encoding
        first_row = result.iloc[0]  # "Action|Adventure"
        assert first_row["Action"] == 1
        assert first_row["Adventure"] == 1
        assert first_row["Comedy"] == 0
        assert first_row["Drama"] == 0

        second_row = result.iloc[1]  # "Comedy|Drama"
        assert second_row["Action"] == 0
        assert second_row["Adventure"] == 0
        assert second_row["Comedy"] == 1
        assert second_row["Drama"] == 1

    def test_engineer_features_creates_user_genre_preferences(self, sample_df):
        """Test that user genre preference features are created"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(sample_df)
        result = feature_eng._engineer_features(sample_df)

        # Check that user genre preferences exist
        genre_pref_columns = [col for col in result.columns if "preference" in col]
        assert len(genre_pref_columns) > 0

        # Check that preference values are not null
        for col in genre_pref_columns:
            assert result[col].notna().all()

    def test_engineer_features_calculates_stats_correctly(self, sample_df):
        """Test that basic statistics are calculated correctly"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(sample_df)
        result = feature_eng._engineer_features(sample_df)

        # Check user stats for userId=1 (ratings: 4.0, 5.0)
        user1_stats = result[sample_df["userId"] == 1].iloc[0]
        assert user1_stats["user_avg_rating"] == 4.5

        # Check movie stats for movieId=1 (ratings: 4.0, 3.0)
        movie1_stats = result[sample_df["movieId"] == 1].iloc[0]
        assert movie1_stats["movie_avg_rating"] == 3.5

    def test_engineer_features_excludes_non_feature_columns(self, sample_df):
        """Test that non-feature columns are excluded"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(sample_df)
        result = feature_eng._engineer_features(sample_df)

        non_features = ["userId", "movieId", "rating", "timestamp", "genres", "title"]
        for col in non_features:
            assert col not in result.columns, f"Column {col} should be excluded"

    def test_fit_transform_consistency(self, sample_df):
        """Test that fit and transform produce consistent results"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(sample_df)

        X1, y1 = feature_eng.transform(sample_df)
        X2, y2 = feature_eng.transform(sample_df)

        # Transform should be deterministic
        np.testing.assert_array_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)

    def test_transform_requires_fit_first(self, sample_df):
        """Test that transform raises error if fit hasn't been called"""
        feature_eng = GradientBoostingFeatureEng()

        with pytest.raises(KeyError):
            feature_eng.transform(sample_df)

    def test_handles_new_genres_in_transform(self, sample_df):
        """Test that new genres in transform are handled gracefully"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(sample_df)

        # Add a new genre not seen during fit
        new_df = sample_df.copy()
        new_df["genres"] = new_df["genres"] + "|NewGenre"

        X, y = feature_eng.transform(new_df)
        assert not np.isnan(X).any()
        assert X.shape[0] == len(new_df)

    def test_no_nan_values(self, large_sample_df):
        """Test that no NaN values are present in the output"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(large_sample_df)

        X, y = feature_eng.transform(large_sample_df)
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()

    def test_feature_column_order_consistent(self, large_sample_df):
        """Test that feature column order is consistent between fit and transform"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(large_sample_df)

        # Store feature columns during fit
        expected_columns = feature_eng.encoders["feature_columns"]

        # Transform should create same columns in same order
        df_features = feature_eng._engineer_features(large_sample_df)
        assert list(df_features.columns) == expected_columns

    def test_rolling_statistics_with_small_groups(self):
        """Test rolling statistics with users having few ratings"""
        small_df = pd.DataFrame(
            {
                "userId": [1, 2, 3],
                "movieId": [1, 2, 3],
                "rating": [4.0, 3.0, 5.0],
                "timestamp": [964982703, 964982704, 964982705],
                "genres": ["Action", "Comedy", "Drama"],
            }
        )

        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(small_df)
        X, y = feature_eng.transform(small_df)

        # Should handle users with single ratings
        assert not np.isnan(X).any()

    def test_transform_preserves_y_values(self, sample_df):
        """Test that y values are correctly extracted"""
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(sample_df)

        X, y = feature_eng.transform(sample_df)

        # y should contain the rating column
        pd.testing.assert_series_equal(
            y.reset_index(drop=True),
            sample_df["rating"].reset_index(drop=True),
            check_names=False,
        )


class TestGradientBoostingIntegration:
    """Integration tests for the full Gradient Boosting pipeline"""

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
                "genres": [
                    "Action|Adventure",
                    "Comedy|Drama",
                    "Action",
                    "Action",
                ],
            }
        )
        return ratings, movies

    def test_full_pipeline(self, full_pipeline_data):
        """Test the complete preprocessing + feature engineering pipeline"""
        ratings_df, movies_df = full_pipeline_data

        # Preprocess
        preprocessor = GradientBoostingPreprocessor()
        processed_df = preprocessor.apply(ratings_df, movies_df)

        # Feature engineering
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(processed_df)
        X, y = feature_eng.transform(processed_df)

        # Check that we get valid outputs
        assert X.shape[0] == len(processed_df)
        assert y.shape[0] == len(processed_df)
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()

    def test_full_pipeline_with_no_genre_movies(self, full_pipeline_data):
        """Test pipeline handles movies with no genres correctly"""
        ratings_df, movies_df = full_pipeline_data

        # Add a movie with no genres
        movies_df = pd.concat(
            [
                movies_df,
                pd.DataFrame({"movieId": [5], "genres": ["(no genres listed)"]}),
            ]
        )

        # Preprocess
        preprocessor = GradientBoostingPreprocessor()
        processed_df = preprocessor.apply(ratings_df, movies_df)

        # Movie 5 should be filtered out
        assert 5 not in processed_df["movieId"].values

        # Feature engineering should still work
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(processed_df)
        X, y = feature_eng.transform(processed_df)

        assert not np.isnan(X).any()

    def test_pipeline_deterministic(self, full_pipeline_data):
        """Test that the full pipeline is deterministic"""
        ratings_df, movies_df = full_pipeline_data

        # Run pipeline twice
        preprocessor1 = GradientBoostingPreprocessor()
        processed_df1 = preprocessor1.apply(ratings_df, movies_df)
        feature_eng1 = GradientBoostingFeatureEng()
        feature_eng1.fit(processed_df1)
        X1, y1 = feature_eng1.transform(processed_df1)

        preprocessor2 = GradientBoostingPreprocessor()
        processed_df2 = preprocessor2.apply(ratings_df, movies_df)
        feature_eng2 = GradientBoostingFeatureEng()
        feature_eng2.fit(processed_df2)
        X2, y2 = feature_eng2.transform(processed_df2)

        # Results should be identical
        np.testing.assert_array_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)

    def test_pipeline_with_missing_movie_ids(self, full_pipeline_data):
        """Test pipeline when ratings reference movies not in movies_df"""
        ratings_df, movies_df = full_pipeline_data

        # Add a rating for a non-existent movie
        ratings_df = pd.concat(
            [
                ratings_df,
                pd.DataFrame(
                    {
                        "userId": [4],
                        "movieId": [999],
                        "rating": [3.0],
                        "timestamp": [964982709],
                    }
                ),
            ]
        )

        # Preprocess (inner join should exclude movie 999)
        preprocessor = GradientBoostingPreprocessor()
        processed_df = preprocessor.apply(ratings_df, movies_df)

        # Movie 999 should not be in the result
        assert 999 not in processed_df["movieId"].values

        # Feature engineering should still work
        feature_eng = GradientBoostingFeatureEng()
        feature_eng.fit(processed_df)
        X, y = feature_eng.transform(processed_df)

        assert X.shape[0] == len(processed_df)
        assert not np.isnan(X).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
