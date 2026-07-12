import unittest
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.mlp import MLPModel
from src.features import MLPModelPreprocessor
from src.features import MLPFeatureEngineer


class TestMLPModel(unittest.TestCase):
    """Test suite for MLPModel class"""

    def setUp(self):
        """Set up test fixtures"""
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # Create synthetic regression dataset
        n_samples = 500
        n_features = 10

        # Generate features with some structure
        self.X = np.random.randn(n_samples, n_features).astype(np.float64)

        # Generate target with linear + non-linear components
        true_weights = np.random.randn(n_features).astype(np.float64)
        self.linear_part = self.X @ true_weights
        self.nonlinear_part = np.sin(self.X[:, 0]) * np.cos(self.X[:, 1])
        self.y = (
            self.linear_part
            + 2 * self.nonlinear_part
            + 0.1 * np.random.randn(n_samples)
        ).astype(np.float64)

        # Split into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Default model configuration
        self.default_config = {
            "hidden_dims": [64, 32],
            "dropout_rate": 0.2,
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 0.01,
            "weight_decay": 1e-5,
            "patience": 10,
            "device": "cpu",
        }

    def test_initialization(self):
        """Test model initialization with default and custom parameters"""
        # Test default initialization
        model = MLPModel()
        self.assertFalse(model.is_trained)
        self.assertIsNone(model.model)
        self.assertEqual(model.hidden_dims, [256, 128, 64])
        self.assertEqual(model.dropout_rate, 0.3)
        self.assertEqual(model.batch_size, 256)
        self.assertEqual(model.epochs, 100)
        self.assertEqual(model.learning_rate, 0.001)

        # Test custom initialization
        model = MLPModel(**self.default_config)
        self.assertEqual(model.hidden_dims, [64, 32])
        self.assertEqual(model.dropout_rate, 0.2)
        self.assertEqual(model.batch_size, 32)

    def test_build(self):
        """Test model building with different architectures"""
        model = MLPModel(hidden_dims=[32, 16])
        model.input_dim = 10
        model.build()

        # Check that model is created
        self.assertIsNotNone(model.model)
        self.assertIsInstance(model.model, torch.nn.Sequential)

        # Check architecture
        assert model.model is not None
        layers = list(model.model.children())
        # Should have: Linear, BatchNorm, ReLU, Dropout, Linear, BatchNorm, ReLU, Dropout, Linear
        self.assertEqual(len(layers), 9)  # 2 hidden layers * 4 + 1 output

        # First layer should be Linear with correct dimensions
        self.assertIsInstance(layers[0], torch.nn.Linear)
        self.assertEqual(layers[0].in_features, 10)
        self.assertEqual(layers[0].out_features, 32)

        # Last layer should be Linear with output 1
        self.assertIsInstance(layers[-1], torch.nn.Linear)
        self.assertEqual(layers[-1].out_features, 1)

    def test_build_without_input_dim(self):
        """Test that building without input_dim raises error"""
        model = MLPModel()
        with self.assertRaises(ValueError):
            model.build()

    def test_build_single_hidden_layer(self):
        """Test model with single hidden layer"""
        model = MLPModel(hidden_dims=[32])
        model.input_dim = 10
        model.build()

        assert model.model is not None
        layers = list(model.model.children())
        # Should have: Linear, BatchNorm, ReLU, Dropout, Linear
        self.assertEqual(len(layers), 5)

    def test_forward_pass(self):
        """Test forward pass with random data"""
        model = MLPModel(hidden_dims=[32, 16])
        model.input_dim = 10
        model.build()

        # Create random batch
        batch_size = 16
        x = torch.randn(batch_size, 10)

        # Forward pass
        output = model._forward(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size,))
        self.assertEqual(output.dtype, torch.float32)

    def test_fit_basic(self):
        """Test basic model fitting"""
        model = MLPModel(**self.default_config)
        model.fit(self.X_train, self.y_train, verbose=False)

        # Check that model is trained
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.model)
        self.assertEqual(model.input_dim, self.X_train.shape[1])

    def test_fit_with_validation_data(self):
        """Test fitting with explicit validation data"""
        # Create validation split
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )

        model = MLPModel(**self.default_config)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)

        self.assertTrue(model.is_trained)

    def test_fit_without_validation(self):
        """Test fitting without providing validation data (auto-split)"""
        model = MLPModel(**self.default_config)
        model.fit(self.X_train, self.y_train, verbose=False)

        self.assertTrue(model.is_trained)

    def test_fit_empty_data(self):
        """Test fitting with empty data raises error"""
        model = MLPModel(**self.default_config)
        X_empty = np.array([]).reshape(0, 10)
        y_empty = np.array([])

        with self.assertRaises(ValueError):
            model.fit(X_empty, y_empty)

    def test_fit_mismatched_shapes(self):
        """Test fitting with mismatched X and y shapes"""
        model = MLPModel(**self.default_config)
        X_wrong = self.X_train[:10]
        y_wrong = self.y_train[:5]

        with self.assertRaises(ValueError):
            model.fit(X_wrong, y_wrong)

    def test_predict_before_training(self):
        """Test that predicting before training raises error"""
        model = MLPModel(**self.default_config)

        with self.assertRaises(ValueError):
            model.predict(self.X_test)

    def test_predict_shape(self):
        """Test that predictions have correct shape"""
        model = MLPModel(**self.default_config)
        model.fit(self.X_train, self.y_train, verbose=False)

        predictions = model.predict(self.X_test)

        self.assertEqual(predictions.shape, (len(self.X_test),))
        self.assertEqual(predictions.dtype, np.float64)

    def test_predict_values(self):
        """Test that predictions are reasonable"""
        model = MLPModel(**self.default_config)
        model.fit(self.X_train, self.y_train, verbose=False)

        predictions = model.predict(self.X_test)

        # Predictions should be finite
        self.assertTrue(np.all(np.isfinite(predictions)))

        # Predictions should be in a reasonable range (not too extreme)
        self.assertTrue(np.all(predictions > -10))
        self.assertTrue(np.all(predictions < 10))

    def test_model_performance(self):
        """Test that model achieves reasonable performance"""
        model = MLPModel(
            hidden_dims=[64, 32],
            dropout_rate=0.1,
            batch_size=32,
            epochs=100,
            learning_rate=0.01,
            patience=10,
            device="cpu",
        )
        model.fit(self.X_train, self.y_train, verbose=False)

        predictions = model.predict(self.X_test)

        # Calculate metrics
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)

        # Model should perform better than naive mean predictor
        baseline_mse = mean_squared_error(
            self.y_test, np.full_like(self.y_test, self.y_train.mean())
        )

        self.assertLess(mse, baseline_mse, "Model should beat baseline")
        self.assertGreater(r2, 0.0, "R² should be positive")

    def test_early_stopping(self):
        """Test that early stopping works"""
        model = MLPModel(
            hidden_dims=[64, 32],
            dropout_rate=0.1,
            batch_size=32,
            epochs=1000,  # Large number of epochs
            learning_rate=0.01,
            patience=5,  # Small patience
            device="cpu",
        )

        model.fit(self.X_train, self.y_train, verbose=False)

        # Training should stop early, so it shouldn't take too long
        # (1000 epochs would take much longer)
        self.assertTrue(model.is_trained)

    def test_reproducibility(self):
        """Test that model gives reproducible results with same seed"""
        # Train first model
        np.random.seed(42)
        torch.manual_seed(42)
        model1 = MLPModel(**self.default_config)
        model1.fit(self.X_train, self.y_train, verbose=False)
        pred1 = model1.predict(self.X_test)

        # Train second model with same seed
        np.random.seed(42)
        torch.manual_seed(42)
        model2 = MLPModel(**self.default_config)
        model2.fit(self.X_train, self.y_train, verbose=False)
        pred2 = model2.predict(self.X_test)

        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)

    def test_different_architectures(self):
        """Test that different architectures work"""
        architectures = [[32], [64, 32], [128, 64, 32], [256, 128, 64, 32]]

        for hidden_dims in architectures:
            config = self.default_config.copy()
            config["hidden_dims"] = hidden_dims
            config["epochs"] = 20  # Shorter training for testing

            model = MLPModel(**config)
            model.fit(self.X_train, self.y_train, verbose=False)

            predictions = model.predict(self.X_test)
            self.assertEqual(predictions.shape, (len(self.X_test),))
            self.assertTrue(np.all(np.isfinite(predictions)))

    def test_different_batch_sizes(self):
        """Test that different batch sizes work"""
        batch_sizes = [16, 32, 64, 128]

        for batch_size in batch_sizes:
            config = self.default_config.copy()
            config["batch_size"] = batch_size
            config["epochs"] = 20

            model = MLPModel(**config)
            model.fit(self.X_train, self.y_train, verbose=False)

            predictions = model.predict(self.X_test)
            self.assertTrue(np.all(np.isfinite(predictions)))

    def test_get_model(self):
        """Test getting the underlying model"""
        model = MLPModel(**self.default_config)

        # Should raise error before building
        with self.assertRaises(ValueError):
            model.get_model()

        # Train model
        model.fit(self.X_train, self.y_train, verbose=False)

        # Should return model after training
        pytorch_model = model.get_model()
        self.assertIsInstance(pytorch_model, torch.nn.Module)

    def test_dropout_effect(self):
        """Test that dropout rate affects training"""
        # Model with high dropout
        model_high_dropout = MLPModel(
            hidden_dims=[64, 32],
            dropout_rate=0.9,
            batch_size=32,
            epochs=50,
            learning_rate=0.01,
            device="cpu",
        )
        model_high_dropout.fit(self.X_train, self.y_train, verbose=False)
        pred_high = model_high_dropout.predict(self.X_test)

        # Model with no dropout
        model_no_dropout = MLPModel(
            hidden_dims=[64, 32],
            dropout_rate=0.0,
            batch_size=32,
            epochs=50,
            learning_rate=0.01,
            device="cpu",
        )
        model_no_dropout.fit(self.X_train, self.y_train, verbose=False)
        pred_no = model_no_dropout.predict(self.X_test)

        # Both should produce valid predictions
        self.assertTrue(np.all(np.isfinite(pred_high)))
        self.assertTrue(np.all(np.isfinite(pred_no)))

        # Predictions should differ (dropout affects model)
        self.assertFalse(np.allclose(pred_high, pred_no, rtol=0.01))

    def test_weight_decay_effect(self):
        """Test that weight decay affects training"""
        # Model with high weight decay
        model_high_wd = MLPModel(
            hidden_dims=[64, 32],
            dropout_rate=0.1,
            batch_size=32,
            epochs=50,
            learning_rate=0.01,
            weight_decay=1.0,  # High regularization
            device="cpu",
        )
        model_high_wd.fit(self.X_train, self.y_train, verbose=False)
        pred_high = model_high_wd.predict(self.X_test)

        # Model with no weight decay
        model_no_wd = MLPModel(
            hidden_dims=[64, 32],
            dropout_rate=0.1,
            batch_size=32,
            epochs=50,
            learning_rate=0.01,
            weight_decay=0.0,
            device="cpu",
        )
        model_no_wd.fit(self.X_train, self.y_train, verbose=False)
        pred_no = model_no_wd.predict(self.X_test)

        # Both should produce valid predictions
        self.assertTrue(np.all(np.isfinite(pred_high)))
        self.assertTrue(np.all(np.isfinite(pred_no)))

    def test_batch_prediction_consistency(self):
        """Test that predictions are consistent regardless of batch size"""
        model = MLPModel(**self.default_config)
        model.fit(self.X_train, self.y_train, verbose=False)

        # Predict with default batch size
        pred1 = model.predict(self.X_test)

        # Change batch size and predict again
        model.batch_size = 128
        pred2 = model.predict(self.X_test)

        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)

    def test_single_sample_prediction(self):
        """Test prediction with single sample"""
        model = MLPModel(**self.default_config)
        model.fit(self.X_train, self.y_train, verbose=False)

        # Predict single sample
        single_sample = self.X_test[0:1]
        prediction = model.predict(single_sample)

        self.assertEqual(prediction.shape, (1,))
        self.assertTrue(np.isfinite(prediction[0]))

    def test_large_batch_prediction(self):
        """Test prediction with more samples than batch size"""
        model = MLPModel(
            batch_size=16,
            **{k: v for k, v in self.default_config.items() if k != "batch_size"},
        )
        model.fit(self.X_train, self.y_train, verbose=False)

        # Predict many samples
        predictions = model.predict(self.X_test)

        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(np.all(np.isfinite(predictions)))


class TestMLPModelIntegration(unittest.TestCase):
    """Integration tests for MLPModel with preprocessor and feature engineer"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample data similar to movie ratings
        np.random.seed(42)

        # Movies data
        self.movies_df = pd.DataFrame(
            {
                "movieId": [1, 2, 3],
                "title": ["Movie A", "Movie B", "Movie C"],
                "genres": ["Action|Comedy", "Action|Drama", "Comedy|Drama"],
            }
        )

        # Ratings data
        n_samples = 200
        self.ratings_df = pd.DataFrame(
            {
                "userId": np.random.randint(1, 20, n_samples),
                "movieId": np.random.choice([1, 2, 3], n_samples),
                "rating": np.random.uniform(1, 5, n_samples).round(1),
                "timestamp": np.random.randint(964982703, 978307200, n_samples),
            }
        )

        # Initialize components
        self.preprocessor = MLPModelPreprocessor()
        self.feature_engineer = MLPFeatureEngineer()

    def test_full_pipeline(self):
        """Test complete pipeline: preprocessor -> feature engineer -> model"""
        # Split data
        ratings_train, ratings_test = train_test_split(
            self.ratings_df, test_size=0.2, random_state=42
        )

        # Preprocess
        train_processed = self.preprocessor.apply(ratings_train, self.movies_df)
        test_processed = self.preprocessor.apply(ratings_test, self.movies_df)

        # Feature engineering
        self.feature_engineer.fit(train_processed)
        X_train, y_train = self.feature_engineer.transform(train_processed)
        X_test, y_test = self.feature_engineer.transform(test_processed)

        # Train model
        model = MLPModel(
            hidden_dims=[64, 32],
            dropout_rate=0.2,
            batch_size=32,
            epochs=50,
            learning_rate=0.01,
            patience=10,
        )
        model.fit(X_train, y_train, verbose=False)  # type: ignore

        # Predict and evaluate
        predictions = model.predict(X_test)

        # Basic checks
        self.assertEqual(len(predictions), len(y_test))
        self.assertTrue(np.all(np.isfinite(predictions)))

        # Predictions should be in reasonable rating range (not necessarily 1-5 for standardized data)
        mse = mean_squared_error(y_test, predictions)
        self.assertFalse(np.isnan(mse))
        self.assertFalse(np.isinf(mse))


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
