import pandas as pd
from pathlib import Path
import yaml
import sys
import pickle
import json
from typing import Dict, Any, Tuple
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
import mlflow
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))


def load_params():
    """Load parameters from params.yaml file."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def load_model(name: str):
    """Load trained model from disk."""
    model_path = Path(f"models/{name}_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded {name} model from {model_path}")
    return model


def load_features(name: str) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load features and target for a specific model."""
    features_path = f"data/features/{name}_features.csv"
    df: pd.DataFrame = pd.read_csv(features_path)

    X: npt.NDArray[np.float64] = np.array(
        df.drop("rating", axis=1).values, dtype=np.float64
    )
    y: npt.NDArray[np.float64] = np.array(df["rating"].values, dtype=np.float64)

    return X, y


def load_features_with_split(
    name: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Load features and split into train/test sets."""
    X, y = load_features(name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def evaluate_model(
    model, X: np.ndarray, y: np.ndarray, prefix: str = ""
) -> Dict[str, float]:
    """Evaluate model performance with 4 key metrics.

    Metrics:
    1. RMSE: Root Mean Squared Error (lower is better)
    2. MAE: Mean Absolute Error (lower is better)
    3. R²: Coefficient of Determination (higher is better, max 1.0)
    4. MAPE: Mean Absolute Percentage Error (lower is better)
    """
    # Predictions
    y_pred = model.predict(X)

    # 1. Root Mean Squared Error (RMSE)
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)

    # 2. Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y - y_pred))

    # 3. R² Score (Coefficient of Determination)
    try:
        r2 = model.get_model().score(X, y)
    except Exception:
        # Fallback calculation
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

    # 4. Mean Absolute Percentage Error (MAPE)
    # Add small epsilon to avoid division by zero
    mape = np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100

    metrics = {
        f"{prefix}rmse": float(rmse),
        f"{prefix}mae": float(mae),
        f"{prefix}r2": float(r2),
        f"{prefix}mape": float(mape),
    }

    return metrics


def setup_mlflow(params: Dict[str, Any]) -> None:
    """Setup MLflow tracking for evaluation."""
    tracking_uri = params.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = params.get("mlflow", {}).get(
        "experiment_name", "movie_rating_prediction"
    )
    mlflow.set_experiment(experiment_name)

    run_name = params.get("mlflow", {}).get(
        "evaluation_run_name", f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    mlflow.start_run(run_name=run_name)
    print(f"MLflow evaluation run started: {run_name}")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {experiment_name}")


def log_evaluation_to_mlflow(
    name: str,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    X_train_shape: Tuple[int, int],
    X_test_shape: Tuple[int, int],
    model_params: Dict[str, Any],
) -> None:
    """Log evaluation metrics to MLflow."""

    # Log model info
    mlflow.log_params(
        {
            f"{name}_model_type": name,
            f"{name}_train_samples": X_train_shape[0],
            f"{name}_train_features": X_train_shape[1],
            f"{name}_test_samples": X_test_shape[0],
            f"{name}_test_features": X_test_shape[1],
        }
    )

    # Log model parameters if available
    for param_name, param_value in model_params.items():
        # Skip non-serializable or non-numeric parameters
        if isinstance(param_value, (int, float, str, bool)):
            mlflow.log_param(f"{name}_{param_name}", param_value)
        elif isinstance(param_value, (list, tuple, dict)):
            # Convert to string for logging
            mlflow.log_param(f"{name}_{param_name}", str(param_value))
        # Skip other types

    # Log metrics
    train_prefixed = {f"{name}_train_{k}": v for k, v in train_metrics.items()}
    test_prefixed = {f"{name}_test_{k}": v for k, v in test_metrics.items()}
    mlflow.log_metrics(train_prefixed)
    mlflow.log_metrics(test_prefixed)

    print(f"Logged {name} evaluation metrics to MLflow")


def print_comparison(all_metrics: Dict[str, Any]) -> None:
    """Print a formatted comparison table of all models."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - Test Performance")
    print("=" * 80)

    # Check if we have any models to compare
    if not all_metrics:
        print("No models to compare.")
        return

    # Define metric display names and formatting
    metrics_to_show = [
        ("rmse", "RMSE", "↓"),
        ("mae", "MAE", "↓"),
        ("r2", "R²", "↑"),
        ("mape", "MAPE%", "↓"),
    ]

    # Find best model for each metric
    best_models = {}
    for metric_key, _, direction in metrics_to_show:
        valid_models = {}
        for name, metrics in all_metrics.items():
            if "test" in metrics and metric_key in metrics["test"]:
                valid_models[name] = metrics["test"][metric_key]

        if valid_models:
            if direction == "↑":  # Higher is better
                best_models[metric_key] = max(valid_models, key=valid_models.get)  # type: ignore
            else:  # Lower is better
                best_models[metric_key] = min(valid_models, key=valid_models.get)  # type: ignore

    # Print header
    header = f"{'Model':<20}"
    for metric_key, display_name, _ in metrics_to_show:
        header += f"{display_name:<12}"
    print(header)
    print("-" * 80)

    # Print each model's metrics
    for name, metrics in all_metrics.items():
        if "error" in metrics or "test" not in metrics:
            continue

        row = f"{name:<20}"
        for metric_key, display_name, direction in metrics_to_show:
            value = metrics["test"].get(metric_key, np.nan)
            if np.isnan(value):
                row += f"{'N/A':<12}"
            else:
                # Format based on metric type
                if metric_key == "mape":
                    formatted = f"{value:.2f}"
                elif metric_key == "r2":
                    formatted = f"{value:.4f}"
                else:
                    formatted = f"{value:.4f}"

                # Highlight best model
                if best_models.get(metric_key) == name:
                    row += f"*{formatted:<11}"
                else:
                    row += f"{formatted:<12}"
        print(row)

    print("=" * 80)
    print("* indicates best performing model for that metric")
    print("↓ = lower is better | ↑ = higher is better")
    print("=" * 80)


def get_model_params(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model parameters from params."""
    param_key = f"{name}_params"
    return params.get(param_key, {})


def serialize_metrics(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert metrics to JSON-serializable format."""
    serializable = {}
    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            serializable[key] = serialize_metrics(value)
        elif isinstance(value, (np.integer, np.floating)):
            serializable[key] = float(value)
        elif isinstance(value, (np.ndarray, list, tuple)):
            serializable[key] = [
                float(x) if isinstance(x, (np.number, float, int)) else str(x)
                for x in value
            ]
        elif isinstance(value, (int, float, str, bool)):
            serializable[key] = value
        elif value is None:
            serializable[key] = None
        else:
            # Convert anything else to string
            serializable[key] = str(value)
    return serializable


def main():
    # Load parameters
    params = load_params()

    # Ensure metrics directory exists
    Path("metrics").mkdir(parents=True, exist_ok=True)

    # Setup MLflow
    setup_mlflow(params)

    # Get split parameters (should match training)
    test_size = params.get("evaluate", {}).get("test_size", 0.2)
    random_state = params.get("evaluate", {}).get("random_state", 42)

    # Track all metrics
    all_metrics = {}

    try:
        # Evaluate models that are enabled
        for name, enabled in params.get("models", {}).items():
            if not enabled:
                print(f"Skipping {name} model (disabled in params)")
                continue

            print(f"\n{'=' * 60}")
            print(f"Evaluating {name} model...")
            print(f"{'=' * 60}")

            try:
                # Check if model exists
                model_path = Path(f"models/{name}_model.pkl")
                if not model_path.exists():
                    print(f"⚠️ Model not found: {model_path}")
                    print("   Run training first: dvc repro train")
                    all_metrics[name] = {"error": "Model not found"}
                    continue

                # Load model
                model = load_model(name)

                # Load and split features
                X_train, X_test, y_train, y_test = load_features_with_split(
                    name, test_size=test_size, random_state=random_state
                )
                print("Data split:")
                print(
                    f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features"
                )
                print(f"  Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")

                # Get model parameters
                model_params = get_model_params(name, params)

                # Evaluate on training data
                train_metrics = evaluate_model(model, X_train, y_train, prefix="")
                print(f"\n{name} Training Performance:")
                for metric_name, value in train_metrics.items():
                    print(f"  {metric_name}: {value:.4f}")

                # Evaluate on test data
                test_metrics = evaluate_model(model, X_test, y_test, prefix="")
                print(f"\n{name} Test Performance:")
                for metric_name, value in test_metrics.items():
                    print(f"  {metric_name}: {value:.4f}")

                # Log to MLflow
                log_evaluation_to_mlflow(
                    name=name,
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                    X_train_shape=X_train.shape,
                    X_test_shape=X_test.shape,
                    model_params=model_params,
                )

                # Store metrics
                all_metrics[name] = {
                    "train": train_metrics,
                    "test": test_metrics,
                    "model_params": model_params,
                    "train_samples": X_train.shape[0],
                    "test_samples": X_test.shape[0],
                }

            except Exception as e:
                print(f"Error evaluating {name} model: {str(e)}")
                all_metrics[name] = {"error": str(e)}
                mlflow.log_param(f"{name}_error", str(e))

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise
    finally:
        # End MLflow run
        mlflow.end_run()
        print("\nMLflow evaluation run ended")

    # Save all metrics - using serialization function
    metrics_path = Path("metrics/evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        serializable_metrics = serialize_metrics(all_metrics)
        json.dump(serializable_metrics, f, indent=2)

    print(f"\nEvaluation metrics saved to {metrics_path}")

    # Print detailed comparison
    print_comparison(all_metrics)

    # Summary
    print(f"\n{'=' * 60}")
    print("Evaluation Summary")
    print(f"{'=' * 60}")

    for name, metrics in all_metrics.items():
        if "error" in metrics:
            print(f"❌ {name}: Failed - {metrics['error']}")
        else:
            test_r2 = metrics.get("test", {}).get("r2", "N/A")
            test_rmse = metrics.get("test", {}).get("rmse", "N/A")
            test_mae = metrics.get("test", {}).get("mae", "N/A")
            test_mape = metrics.get("test", {}).get("mape", "N/A")
            train_samples = metrics.get("train_samples", "N/A")
            test_samples = metrics.get("test_samples", "N/A")
            print(f"✅ {name}:")
            print(f"   R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}")
            print(f"   MAE = {test_mae:.4f}, MAPE = {test_mape:.2f}%")
            print(f"   ({train_samples} train, {test_samples} test)")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
