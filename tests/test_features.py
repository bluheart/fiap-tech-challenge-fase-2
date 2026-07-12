import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features import MLPModelPreprocessor, MLPFeatureEngineer


class TestMLPModelPreprocessor(unittest.TestCase):
    """Test suite for MLPModelPreprocessor"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample movies data
        self.movies_df = pd.DataFrame(
            {
                "movieId": [1, 2, 3, 4, 5],
                "title": [
                    "Toy Story (1995)",
                    "Jumanji (1995)",
                    "No Genre Movie (1995)",
                    "Deadpool (2016)",
                    "Matrix, The (1999)",
                ],
                "genres": [
                    "Adventure|Animation|Children|Comedy|Fantasy",
                    "Adventure|Children|Fantasy",
                    "(no genres listed)",
                    "Action|Comedy",
                    "Action|Sci-Fi",
                ],
            }
        )

        # Create sample ratings data
        self.ratings_df = pd.DataFrame(
            {
                "userId": [1, 1, 2, 2, 3],
                "movieId": [1, 2, 2, 4, 1],
                "rating": [4.0, 3.5, 5.0, 2.0, 4.5],
                "timestamp": [964982703, 964982931, 964983194, 964983445, 964983674],
            }
        )

        self.preprocessor = MLPModelPreprocessor()

    def test_removes_no_genres_listed(self):
        """Test that movies with '(no genres listed)' are removed"""
        result = self.preprocessor.apply(self.ratings_df, self.movies_df)

        # Movie 3 has no genres listed and should not appear
        self.assertNotIn(3, result["movieId"].values)

        # Other movies should still be present
        self.assertIn(1, result["movieId"].values)
        self.assertIn(2, result["movieId"].values)
        self.assertIn(4, result["movieId"].values)

    def test_merges_genres_correctly(self):
        """Test that genres are properly merged from movies to ratings"""
        result = self.preprocessor.apply(self.ratings_df, self.movies_df)

        # Check that genres column exists
        self.assertIn("genres", result.columns)

        # Check specific genre assignments
        movie_1_ratings = result[result["movieId"] == 1]
        self.assertTrue(
            all(
                movie_1_ratings["genres"]
                == "Adventure|Animation|Children|Comedy|Fantasy"
            )
        )

        movie_2_ratings = result[result["movieId"] == 2]
        self.assertTrue(all(movie_2_ratings["genres"] == "Adventure|Children|Fantasy"))

    def test_preserves_original_columns(self):
        """Test that original columns are preserved"""
        result = self.preprocessor.apply(self.ratings_df, self.movies_df)

        expected_columns = ["userId", "movieId", "rating", "timestamp", "genres"]
        for col in expected_columns:
            self.assertIn(col, result.columns)

    def test_handles_empty_ratings(self):
        """Test behavior with empty ratings dataframe"""
        empty_ratings = pd.DataFrame(
            columns=["userId", "movieId", "rating", "timestamp"]
        )
        result = self.preprocessor.apply(empty_ratings, self.movies_df)

        self.assertEqual(len(result), 0)
        self.assertIn("genres", result.columns)

    def test_handles_missing_columns(self):
        """Test that missing required columns raise KeyError"""
        bad_movies = self.movies_df.drop("genres", axis=1)

        with self.assertRaises(KeyError):
            self.preprocessor.apply(self.ratings_df, bad_movies)

    def test_inner_join_removes_unmatched_movies(self):
        """Test that ratings for movies not in movies_df are removed"""
        ratings_with_extra = pd.concat(
            [
                self.ratings_df,
                pd.DataFrame(
                    {
                        "userId": [4],
                        "movieId": [99],  # Non-existent movie
                        "rating": [3.0],
                        "timestamp": [964983999],
                    }
                ),
            ]
        )

        result = self.preprocessor.apply(ratings_with_extra, self.movies_df)
        self.assertNotIn(99, result["movieId"].values)

    def test_does_not_modify_original_dataframes(self):
        """Test that original dataframes are not modified"""
        movies_original = self.movies_df.copy()
        ratings_original = self.ratings_df.copy()

        _ = self.preprocessor.apply(self.ratings_df, self.movies_df)

        pd.testing.assert_frame_equal(self.movies_df, movies_original)
        pd.testing.assert_frame_equal(self.ratings_df, ratings_original)


class TestMLPFeatureEngineer(unittest.TestCase):
    """Test suite for MLPFeatureEngineer"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample movies data
        self.movies_df = pd.DataFrame(
            {
                "movieId": [1, 2, 3],
                "title": ["Movie A", "Movie B", "Movie C"],
                "genres": ["Action|Comedy", "Action|Drama", "Comedy|Drama"],
            }
        )

        # Create larger sample ratings for better statistics
        np.random.seed(42)
        n_samples = 200
        self.ratings_train = pd.DataFrame(
            {
                "userId": np.random.randint(1, 11, n_samples),
                "movieId": np.random.choice([1, 2, 3], n_samples),
                "rating": np.random.uniform(1, 5, n_samples).round(1),
                "timestamp": np.random.randint(964982703, 978307200, n_samples),
            }
        )

        self.ratings_test = pd.DataFrame(
            {
                "userId": [1, 2, 5],
                "movieId": [1, 3, 2],
                "rating": [4.0, 3.5, 2.5],
                "timestamp": [978307200, 978307500, 978307800],
            }
        )

        # Preprocess data
        self.preprocessor = MLPModelPreprocessor()
        self.train_processed = self.preprocessor.apply(
            self.ratings_train, self.movies_df
        )
        self.test_processed = self.preprocessor.apply(self.ratings_test, self.movies_df)

        # Create feature engineer
        self.feature_engineer = MLPFeatureEngineer()

    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining"""
        result = self.feature_engineer.fit(self.train_processed)
        self.assertIsInstance(result, MLPFeatureEngineer)
        self.assertEqual(result, self.feature_engineer)

    def test_fit_stores_encoders(self):
        """Test that fit stores necessary encoders"""
        self.feature_engineer.fit(self.train_processed)

        # Check that all required encoders are stored
        self.assertIn("user_stats", self.feature_engineer.encoders)
        self.assertIn("movie_stats", self.feature_engineer.encoders)
        self.assertIn("genre_columns", self.feature_engineer.encoders)
        self.assertIn("scaler", self.feature_engineer.encoders)

    def test_transform_requires_fit(self):
        """Test that transform raises error if not fitted"""
        with self.assertRaises(ValueError):
            self.feature_engineer.transform(self.train_processed)

    def test_transform_output_shapes(self):
        """Test that transform returns correct shapes"""
        self.feature_engineer.fit(self.train_processed)
        X, y = self.feature_engineer.transform(self.test_processed)

        # Check types
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, pd.Series)

        # Check shapes
        self.assertEqual(X.shape[0], len(self.test_processed))
        self.assertEqual(len(y), len(self.test_processed))

        # X should have multiple features
        self.assertGreater(X.shape[1], 0)

    def test_movie_statistics_features(self):
        """Test that movie statistics features are created"""
        self.feature_engineer.fit(self.train_processed)
        X, _ = self.feature_engineer.transform(self.test_processed)

        feature_cols = self.feature_engineer.feature_columns
        self.assertIn("movie_avg_rating", feature_cols)
        self.assertIn("movie_std_rating", feature_cols)
        self.assertIn("movie_rating_count", feature_cols)

    def test_genre_features(self):
        """Test that genre one-hot encoding features are created"""
        self.feature_engineer.fit(self.train_processed)

        feature_cols = self.feature_engineer.feature_columns
        # Should have genre columns
        genre_cols = [
            col for col in feature_cols if col in ["Action", "Comedy", "Drama"]
        ]
        self.assertGreater(len(genre_cols), 0)

        X, _ = self.feature_engineer.transform(self.test_processed)

        # Genre features should be binary or scaled versions
        action_idx = feature_cols.index("Action")
        comedy_idx = feature_cols.index("Comedy")

        # Check that Action and Comedy are present for movie 1
        movie_1_sample = self.test_processed["movieId"] == 1
        if movie_1_sample.any():
            self.assertNotEqual(X[movie_1_sample, action_idx][0], 0)
            self.assertNotEqual(X[movie_1_sample, comedy_idx][0], 0)

    def test_time_features(self):
        """Test that time-based features are created"""
        self.feature_engineer.fit(self.train_processed)

        feature_cols = self.feature_engineer.feature_columns
        self.assertIn("rating_hour", feature_cols)
        self.assertIn("rating_dayofweek", feature_cols)

        X, _ = self.feature_engineer.transform(self.test_processed)

        # Hour should be between 0-23 (before scaling it might not be exactly)
        hour_idx = feature_cols.index("rating_hour")
        # After scaling, values should still be finite
        self.assertTrue(np.all(np.isfinite(X[:, hour_idx])))

    def test_interaction_features(self):
        """Test that interaction features are created"""
        self.feature_engineer.fit(self.train_processed)

        feature_cols = self.feature_engineer.feature_columns
        self.assertIn("user_movie_avg_interaction", feature_cols)

        X, _ = self.feature_engineer.transform(self.test_processed)

        # Interaction feature should exist and have finite values
        interaction_idx = feature_cols.index("user_movie_avg_interaction")
        self.assertTrue(np.all(np.isfinite(X[:, interaction_idx])))

    def test_handles_new_users_in_test(self):
        """Test that feature engineering handles users not seen in training"""
        # Add a completely new user to test data
        new_user_test = pd.concat(
            [
                self.test_processed,
                pd.DataFrame(
                    {
                        "userId": [99],
                        "movieId": [1],
                        "rating": [3.0],
                        "timestamp": [978308000],
                        "genres": ["Action|Comedy"],
                    }
                ),
            ]
        )

        self.feature_engineer.fit(self.train_processed)
        X, y = self.feature_engineer.transform(new_user_test)

        # Should not crash and should return valid data
        self.assertEqual(X.shape[0], len(new_user_test))
        self.assertTrue(np.all(np.isfinite(X)))

    def test_handles_new_movies_in_test(self):
        """Test that feature engineering handles movies not seen in training"""
        # Add a new movie to test data
        new_movie_test = pd.concat(
            [
                self.test_processed,
                pd.DataFrame(
                    {
                        "userId": [1],
                        "movieId": [99],  # New movie
                        "rating": [3.0],
                        "timestamp": [978308000],
                        "genres": ["Horror|Thriller"],
                    }
                ),
            ]
        )

        self.feature_engineer.fit(self.train_processed)
        X, y = self.feature_engineer.transform(new_movie_test)

        # Should not crash (new genres will be set to 0)
        self.assertEqual(X.shape[0], len(new_movie_test))
        self.assertTrue(np.all(np.isfinite(X)))

    def test_feature_scaling(self):
        """Test that features are properly standardized"""
        self.feature_engineer.fit(self.train_processed)
        X_train, _ = self.feature_engineer.transform(self.train_processed)

        # Training features should have mean ≈ 0 and std ≈ 1
        means = np.mean(X_train, axis=0)
        stds = np.std(X_train, axis=0)

        # Allow some tolerance due to floating point
        self.assertTrue(np.all(np.abs(means) < 0.1))
        self.assertTrue(np.all(np.abs(stds - 1.0) < 0.1))

    def test_consistent_feature_order(self):
        """Test that feature order is consistent between fit and transform"""
        self.feature_engineer.fit(self.train_processed)

        # Transform twice and check feature order is the same
        X1, _ = self.feature_engineer.transform(self.train_processed.head(10))
        X2, _ = self.feature_engineer.transform(self.train_processed.head(10))

        np.testing.assert_array_almost_equal(X1, X2)

    def test_target_variable_preserved(self):
        """Test that target variable (rating) is correctly returned"""
        self.feature_engineer.fit(self.train_processed)
        X, y = self.feature_engineer.transform(self.test_processed)

        # Target should match original ratings
        pd.testing.assert_series_equal(
            y.reset_index(drop=True),
            self.test_processed["rating"].reset_index(drop=True),
        )

    def test_no_data_leakage(self):
        """Test that transform doesn't use test data statistics"""
        self.feature_engineer.fit(self.train_processed)
        X, _ = self.feature_engineer.transform(self.test_processed)

        # Transform again with different test data
        X2, _ = self.feature_engineer.transform(self.train_processed.head(10))

        # The transformation should be consistent (same scaler applied)
        self.assertEqual(X.shape[1], X2.shape[1])

    def test_fit_transform_idempotent(self):
        """Test that fitting and transforming training data is consistent"""
        self.feature_engineer.fit(self.train_processed)
        X1, y1 = self.feature_engineer.transform(self.train_processed)

        # Fit again and transform
        feature_engineer2 = MLPFeatureEngineer()
        feature_engineer2.fit(self.train_processed)
        X2, y2 = feature_engineer2.transform(self.train_processed)

        # Results should be the same
        np.testing.assert_array_almost_equal(X1, X2)
        pd.testing.assert_series_equal(
            y1.reset_index(drop=True), y2.reset_index(drop=True)
        )

    def test_large_dataset_performance(self):
        """Test performance with larger dataset"""
        # Create larger dataset
        n_samples = 1000
        large_train = pd.DataFrame(
            {
                "userId": np.random.randint(1, 51, n_samples),
                "movieId": np.random.choice([1, 2, 3], n_samples),
                "rating": np.random.uniform(1, 5, n_samples).round(1),
                "timestamp": np.random.randint(964982703, 978307200, n_samples),
                "genres": np.random.choice(
                    ["Action|Comedy", "Action|Drama", "Comedy|Drama"], n_samples
                ),
            }
        )

        import time

        start_time = time.time()

        self.feature_engineer.fit(large_train)
        X, y = self.feature_engineer.transform(large_train)

        elapsed_time = time.time() - start_time

        # Should complete within reasonable time (e.g., 5 seconds)
        self.assertLess(elapsed_time, 5.0)
        self.assertEqual(X.shape[0], n_samples)


class TestIntegration(unittest.TestCase):
    """Integration tests for preprocessor and feature engineer together"""

    def setUp(self):
        """Set up test fixtures"""
        self.movies_df = pd.DataFrame(
            {
                "movieId": [1, 2, 3],
                "title": ["Movie A", "Movie B", "Movie C"],
                "genres": ["Action|Comedy", "Action|Drama", "Comedy|Sci-Fi"],
            }
        )

        np.random.seed(42)
        self.ratings_df = pd.DataFrame(
            {
                "userId": np.random.randint(1, 20, 500),
                "movieId": np.random.choice([1, 2, 3], 500),
                "rating": np.random.uniform(1, 5, 500).round(1),
                "timestamp": np.random.randint(964982703, 978307200, 500),
            }
        )

        self.preprocessor = MLPModelPreprocessor()
        self.feature_engineer = MLPFeatureEngineer()

    def test_full_pipeline(self):
        """Test the complete preprocessing and feature engineering pipeline"""
        # Split data
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(
            self.ratings_df, test_size=0.2, random_state=42
        )

        # Apply preprocessor
        train_processed = self.preprocessor.apply(train_df, self.movies_df)
        test_processed = self.preprocessor.apply(test_df, self.movies_df)

        # Fit and transform
        self.feature_engineer.fit(train_processed)
        X_train, y_train = self.feature_engineer.transform(train_processed)
        X_test, y_test = self.feature_engineer.transform(test_processed)

        # Basic checks
        self.assertEqual(X_train.shape[0], len(train_processed))
        self.assertEqual(X_test.shape[0], len(test_processed))
        self.assertEqual(len(y_train), len(train_processed))
        self.assertEqual(len(y_test), len(test_processed))

        # Check that features are finite and well-scaled
        self.assertTrue(np.all(np.isfinite(X_train)))
        self.assertTrue(np.all(np.isfinite(X_test)))

        # Check feature consistency
        self.assertEqual(X_train.shape[1], X_test.shape[1])


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
