# src/predict_mlp.py
"""
MLP Model Prediction Module.

This module provides a predictor class for making predictions using trained
MLP models on movie rating data.
"""

import pandas as pd
import numpy as np
import joblib
import torch
from torch import nn
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, Any, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SimpleMLPPredictor:
    """
    Simple predictor for MLP model.

    This class handles loading a trained MLP model, preprocessing data,
    making predictions, and saving results. It automatically manages
    feature engineering and data preprocessing.

    Attributes:
        model_path (Path): Path to the saved MLP model file.
        model (Optional[Union[nn.Module, Any]]): Loaded MLP model.
        feature_engineer (Optional[Any]): Loaded feature engineer.
        device (str): Device to run predictions on ('cuda' or 'cpu').

    Example:
        >>> predictor = SimpleMLPPredictor()
        >>> results = predictor.predict_and_save(
        ...     ratings_path='data/raw/ratings.csv',
        ...     movies_path='data/raw/movies.csv',
        ...     output_path='predictions.csv'
        ... )
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the MLP predictor.

        Args:
            model_path: Path to the MLP model file. If None, uses default
                path 'models/multi_layer_perceptron_model.pkl'.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if model_path is None:
            self.model_path: Path = (
                project_root / "models" / "multi_layer_perceptron_model.pkl"
            )
        else:
            self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"MLP model not found at: {self.model_path}")

        self.model: Optional[Union[nn.Module, Any]] = None
        self.feature_engineer: Optional[Any] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_model()

    def _load_model(self) -> None:
        """
        Load the MLP model and feature engineer from disk.

        This method attempts to load both the model and its associated
        feature engineer. If the feature engineer is not found, it will
        be created during the prediction process.

        Raises:
            Exception: If the model fails to load.
        """
        print(f"Loading MLP model from: {self.model_path}")
        print(f"Using device: {self.device}")

        # Load the saved data
        saved_data: Dict[str, Any] = joblib.load(self.model_path)

        # Extract model and feature engineer
        if isinstance(saved_data, dict):
            self.model = saved_data.get("model")
            self.feature_engineer = saved_data.get("feature_engineer")
        else:
            self.model = saved_data
            # Try to load feature engineer separately
            feat_path: Path = (
                self.model_path.parent / "multi_layer_perceptron_feature_engineer.pkl"
            )
            if feat_path.exists():
                self.feature_engineer = joblib.load(feat_path)

        # Set model to evaluation mode and move to device
        if self.model and hasattr(self.model, "eval"):
            self.model.eval()
            if hasattr(self.model, "to"):
                self.model.to(self.device)

        print("✓ Model loaded successfully")

    def preprocess_data(
        self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Preprocess ratings and movies data.

        This method applies the MLP preprocessing pipeline, which filters
        out movies with no genres and merges ratings with movie data.

        Args:
            ratings_df: DataFrame containing user ratings with columns
                including 'userId', 'movieId', 'timestamp', and 'rating'.
            movies_df: DataFrame containing movie metadata with columns
                including 'movieId' and 'genres'.

        Returns:
            A tuple containing:
                - processed_df: Preprocessed DataFrame ready for feature engineering
                - kept_indices: Array of original indices that were kept after filtering

        Raises:
            ImportError: If the preprocessor module cannot be imported.
        """
        from src.features.preprocessor import MLPModelPreprocessor

        preprocessor = MLPModelPreprocessor()

        # Apply preprocessing
        processed_df: pd.DataFrame = preprocessor.apply(ratings_df, movies_df)

        # Get indices of rows that were kept (after filtering)
        # We need to align predictions with original data
        # Create a merge key to track original rows
        original_df: pd.DataFrame = ratings_df.copy()
        original_df["_original_index"] = original_df.index

        # Merge to find which rows were kept
        merged: pd.DataFrame = processed_df.merge(
            original_df[["userId", "movieId", "timestamp", "_original_index"]],
            on=["userId", "movieId", "timestamp"],
            how="left",
        )

        kept_indices: np.ndarray = merged["_original_index"].values  # type: ignore

        return processed_df, kept_indices

    def predict(
        self, data_df: pd.DataFrame, movies_df: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with the MLP model.

        This method handles the full prediction pipeline including
        preprocessing, feature engineering, and model inference.

        Args:
            data_df: DataFrame with ratings data. Must contain columns
                'userId', 'movieId', 'rating', and 'timestamp'.
            movies_df: Optional DataFrame with movies data. Must contain
                columns 'movieId' and 'genres'. Required if data_df is
                not already preprocessed.

        Returns:
            A tuple containing:
                - predictions: Array of predicted ratings for kept rows
                - kept_indices: Indices of original rows that were kept

        Raises:
            ValueError: If the model hasn't been loaded.
            ImportError: If required modules cannot be imported.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call _load_model() first.")

        # Preprocess if movies_df provided
        if movies_df is not None:
            processed_df, kept_indices = self.preprocess_data(data_df, movies_df)
        else:
            processed_df = data_df
            kept_indices = np.arange(len(data_df))

        # Create feature engineer if not available
        if self.feature_engineer is None:
            from src.features.feature_engineering import MLPFeatureEngineer

            print("Creating new feature engineer...")
            self.feature_engineer = MLPFeatureEngineer()
            self.feature_engineer.fit(processed_df)

        # Transform features
        X: np.ndarray
        X, _ = self.feature_engineer.transform(processed_df)
        print(f"Feature matrix shape: {X.shape}")

        # Convert to PyTorch tensor
        X_tensor: torch.Tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Make predictions
        with torch.no_grad():
            if hasattr(self.model, "predict"):
                # Scikit-learn style model
                predictions: np.ndarray = self.model.predict(X)  # type: ignore
            else:
                # PyTorch model
                predictions = self.model(X_tensor).cpu().numpy().flatten()  # type: ignore

        # Clip to valid rating range
        predictions = np.clip(predictions, 0.5, 5.0)

        return predictions, kept_indices

    def predict_and_save(
        self,
        ratings_path: str,
        movies_path: str,
        output_path: Optional[str] = None,
        add_actual_rating: bool = True,
    ) -> pd.DataFrame:
        """
        Load data, make predictions, and save results.

        This is the main entry point for making predictions. It handles
        loading data, preprocessing, prediction, and saving results.

        Args:
            ratings_path: Path to the ratings CSV file.
            movies_path: Path to the movies CSV file.
            output_path: Optional path to save predictions. If None,
                predictions are saved to 'predictions/mlp_predictions.csv'.
            add_actual_rating: Whether to include actual ratings in the
                output if available.

        Returns:
            DataFrame containing predictions with identifiers and optionally
            actual ratings and error metrics.

        Raises:
            FileNotFoundError: If input files do not exist.
            Exception: If prediction fails.
        """
        # Load data
        ratings_path_resolved: Path = Path(ratings_path)
        movies_path_resolved: Path = Path(movies_path)

        if not ratings_path_resolved.exists():
            ratings_path_resolved = project_root / ratings_path

        if not movies_path_resolved.exists():
            movies_path_resolved = project_root / movies_path

        print(f"Loading ratings from: {ratings_path_resolved}")
        ratings_df: pd.DataFrame = pd.read_csv(ratings_path_resolved)

        print(f"Loading movies from: {movies_path_resolved}")
        movies_df: pd.DataFrame = pd.read_csv(movies_path_resolved)

        print(f"Loaded {len(ratings_df)} ratings and {len(movies_df)} movies")

        # Make predictions
        predictions: np.ndarray
        kept_indices: np.ndarray
        predictions, kept_indices = self.predict(ratings_df, movies_df)

        # Create results DataFrame with original data for kept rows
        kept_ratings: pd.DataFrame = ratings_df.iloc[kept_indices].reset_index(
            drop=True
        )

        results: pd.DataFrame = pd.DataFrame({"predicted_rating": predictions})

        # Add identifiers if available
        if "userId" in kept_ratings.columns:
            results["userId"] = kept_ratings["userId"].values
        if "movieId" in kept_ratings.columns:
            results["movieId"] = kept_ratings["movieId"].values
        if "timestamp" in kept_ratings.columns:
            results["timestamp"] = kept_ratings["timestamp"].values

        # Add actual ratings if available
        if add_actual_rating and "rating" in kept_ratings.columns:
            results["actual_rating"] = kept_ratings["rating"].values
            results["error"] = np.abs(predictions - kept_ratings["rating"].values)

        # Add original index to track which rows were kept
        results["original_index"] = kept_indices

        # Save if output path provided
        output_path_resolved: Path
        if output_path:
            output_path_resolved = Path(output_path)
        else:
            output_path_resolved = Path("predictions/mlp_predictions.csv")

        output_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path_resolved, index=False)
        print(f"✓ Predictions saved to: {output_path_resolved}")

        return results


def main() -> None:
    """
    Command line entry point for MLP prediction.

    This function parses command line arguments and runs the prediction
    pipeline using the MLP model.
    """
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Make predictions using the MLP model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict with default paths
  python predict_mlp.py --ratings data/raw/ratings.csv --movies data/raw/movies.csv

  # Predict and save to custom location
  python predict_mlp.py --ratings data/raw/ratings.csv --movies data/raw/movies.csv --output my_predictions.csv

  # Use specific model file
  python predict_mlp.py --ratings data/raw/ratings.csv --movies data/raw/movies.csv --model models/mlp_model.pkl
        """,
    )

    parser.add_argument(
        "--ratings", type=str, required=True, help="Path to ratings CSV file"
    )

    parser.add_argument(
        "--movies", type=str, required=True, help="Path to movies CSV file"
    )

    parser.add_argument(
        "--model", type=str, default=None, help="Path to MLP model file (optional)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="predictions/mlp_predictions.csv",
        help="Path to save predictions (default: predictions/mlp_predictions.csv)",
    )

    parser.add_argument(
        "--no-actual",
        action="store_true",
        help="Do not include actual ratings in output",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of predictions to display (default: 10)",
    )

    args: argparse.Namespace = parser.parse_args()

    try:
        # Create predictor
        predictor: SimpleMLPPredictor = SimpleMLPPredictor(model_path=args.model)

        # Make predictions
        results: pd.DataFrame = predictor.predict_and_save(
            ratings_path=args.ratings,
            movies_path=args.movies,
            output_path=args.output,
            add_actual_rating=not args.no_actual,
        )

        # Display summary
        print("\n" + "=" * 60)
        print("MLP PREDICTION SUMMARY")
        print("=" * 60)

        # Count how many rows were filtered out
        original_rows: int = pd.read_csv(args.ratings).shape[0]
        kept_rows: int = len(results)
        filtered_rows: int = original_rows - kept_rows

        print(f"Original ratings: {original_rows}")
        print(f"Kept ratings: {kept_rows}")
        if filtered_rows > 0:
            print(f"Filtered out: {filtered_rows} (movies with no genres)")
        print(f"Total predictions: {len(results)}")
        print("\nPrediction statistics:")
        print(f"  Mean: {results['predicted_rating'].mean():.3f}")
        print(f"  Std: {results['predicted_rating'].std():.3f}")
        print(f"  Min: {results['predicted_rating'].min():.3f}")
        print(f"  Max: {results['predicted_rating'].max():.3f}")
        print(f"  Median: {results['predicted_rating'].median():.3f}")

        if "actual_rating" in results.columns:
            print("\nActual ratings statistics:")
            print(f"  Mean: {results['actual_rating'].mean():.3f}")
            print(f"  Std: {results['actual_rating'].std():.3f}")
            print(f"  Min: {results['actual_rating'].min():.3f}")
            print(f"  Max: {results['actual_rating'].max():.3f}")
            print(f"\nMean absolute error: {results['error'].mean():.3f}")
            print(f"RMSE: {np.sqrt((results['error'] ** 2).mean()):.3f}")

        # Show sample predictions
        display_count: int = min(args.top_n, len(results))
        print(f"\nFirst {display_count} predictions:")
        display_cols: list = ["predicted_rating"]
        if "userId" in results.columns:
            display_cols.insert(0, "userId")
        if "movieId" in results.columns:
            display_cols.insert(1, "movieId")
        if "actual_rating" in results.columns:
            display_cols.append("actual_rating")
            display_cols.append("error")
        if "original_index" in results.columns:
            display_cols.append("original_index")

        print(results[display_cols].head(display_count).to_string(index=False))
        print("\n" + "=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
