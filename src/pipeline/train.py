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

from src.models.factory import ModelFactory


def load_params():
    """Load parameters from params.yaml file."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


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


def save_model(model, name: str) -> Path:
    """Save trained model to disk and return the path."""
    model_path = Path(f"models/{name}_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {name} model to {model_path}")
    return model_path


def create_model_with_factory(name: str, params: Dict[str, Any]):
    """Create model instance using ModelFactory."""
    name_mapping = {
        "linear_regression": "linear_regression",
        "random_forest": "random_forest",
        "gradient_boosting": "gradient_boosting",
        "multi_layer_perceptron": "multi_layer_perceptron",
    }

    model_type = name_mapping.get(name)
    if model_type is None:
        raise ValueError(
            f"Unknown model type: {name}. Available: {list(name_mapping.keys())}"
        )

    model_params_key = f"{name}_params"
    model_params = params.get(model_params_key, {})

    return ModelFactory.create_model(model_type, **model_params)


def setup_mlflow(params: Dict[str, Any]) -> None:
    """Setup MLflow tracking."""
    mlflow_tracking_uri = params.get("mlflow", {}).get("tracking_uri", "file:./mlruns")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    experiment_name = params.get("mlflow", {}).get(
        "experiment_name", "movie_rating_prediction"
    )
    mlflow.set_experiment(experiment_name)

    run_name = params.get("mlflow", {}).get(
        "run_name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    mlflow.start_run(run_name=run_name)
    print(f"MLflow run started: {run_name}")
    print(f"MLflow tracking URI: {mlflow_tracking_uri}")
    print(f"MLflow experiment: {experiment_name}")


def log_model_to_mlflow(
    model,
    name: str,
    params: Dict[str, Any],
    X_train_shape: Tuple[int, int],
    X_test_shape: Tuple[int, int],
    model_params: Dict[str, Any],
    model_path: Path,
) -> None:
    """Log model as pickle artifact and parameters to MLflow."""

    # Log general parameters with model-specific prefix
    mlflow.log_params(
        {
            f"{name}_model_type": name,
            f"{name}_test_size": params.get("train", {}).get("test_size", 0.2),
            f"{name}_random_state": params.get("train", {}).get("random_state", 42),
        }
    )

    # Log individual model parameters
    for param_name, param_value in model_params.items():
        if param_name not in ["models", "train", "mlflow"]:
            if isinstance(param_value, (int, float, str, bool)):
                mlflow.log_param(f"{name}_{param_name}", param_value)
            else:
                mlflow.log_param(f"{name}_{param_name}", str(param_value))

    # Log dataset info
    mlflow.log_params(
        {
            f"{name}_train_samples": X_train_shape[0],
            f"{name}_train_features": X_train_shape[1],
            f"{name}_test_samples": X_test_shape[0],
            f"{name}_test_features": X_test_shape[1],
        }
    )

    # Log the model pickle file as an artifact
    try:
        # Log the pickle file as an artifact
        mlflow.log_artifact(str(model_path), artifact_path="models")
        print(f"Logged {name} model pickle as artifact to MLflow")

        # Also try to log as sklearn model for better MLflow UI integration
        try:
            mlflow.pyfunc.log_model(
                python_model=model.get_model(),
                artifact_path=f"sklearn_models/{name}",
                registered_model_name=f"{name.replace('_', '-')}_model",
            )
            print(f"Logged {name} model as sklearn model to MLflow")
        except Exception as e:
            print(f"Warning: Could not log sklearn model: {e}")
            # Continue with artifact logging only

    except Exception as e:
        print(f"Error logging {name} model to MLflow: {e}")
        # Try alternative: log the entire models directory
        try:
            mlflow.log_artifacts("models", artifact_path="models")
            print("Logged all models directory as artifacts")
        except Exception as e2:
            print(f"Error logging models directory: {e2}")


def main():
    # Load parameters
    params = load_params()

    # Ensure directories exist
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(parents=True, exist_ok=True)

    # Setup MLflow
    setup_mlflow(params)

    # Get split parameters
    test_size = params.get("train", {}).get("test_size", 0.2)
    random_state = params.get("train", {}).get("random_state", 42)

    # Track training info
    training_info = {}
    model_paths = {}  # Store model paths for artifact logging

    try:
        # Train models that are enabled
        for name, enabled in params.get("models", {}).items():
            if not enabled:
                print(f"Skipping {name} model (disabled in params)")
                continue

            print(f"\n{'=' * 60}")
            print(f"Training {name} model...")
            print(f"{'=' * 60}")

            try:
                # Load and split features
                X_train, X_test, y_train, y_test = load_features_with_split(
                    name, test_size=test_size, random_state=random_state
                )
                print("Data split:")
                print(
                    f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features"
                )
                print(f"  Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")

                # Create model using factory
                model = create_model_with_factory(name, params)
                print(f"Created {name} model using ModelFactory")

                # Get model parameters
                model_params = params.get(f"{name}_params", {})

                # Train model
                model.fit(X_train, y_train)
                print("Model trained successfully")

                # Save model and get the path
                model_path = save_model(model, name)
                model_paths[name] = model_path

                # Log to MLflow - with model pickle as artifact
                log_model_to_mlflow(
                    model=model,
                    name=name,
                    params=params,
                    X_train_shape=X_train.shape,
                    X_test_shape=X_test.shape,
                    model_params=model_params,
                    model_path=model_path,
                )

                # Store training info
                training_info[name] = {
                    "train_samples": X_train.shape[0],
                    "test_samples": X_test.shape[0],
                    "train_features": X_train.shape[1],
                    "model_params": model_params,
                    "model_path": str(model_path),
                    "status": "success",
                }

            except Exception as e:
                print(f"Error training {name} model: {str(e)}")
                training_info[name] = {"status": "failed", "error": str(e)}
                mlflow.log_param(f"{name}_error", str(e))

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    finally:
        # End MLflow run
        mlflow.end_run()
        print("\nMLflow run ended")

    # Save training info
    info_path = Path("metrics/train_metrics.json")
    with open(info_path, "w") as f:
        json.dump(training_info, f, indent=2)

    print(f"\nTraining info saved to {info_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Training Summary")
    print(f"{'=' * 60}")

    for name, info in training_info.items():
        if info.get("status") == "failed":
            print(f"❌ {name}: Failed - {info.get('error', 'Unknown error')}")
        else:
            print(f"✅ {name}:")
            print(f"   Train samples: {info['train_samples']}")
            print(f"   Test samples: {info['test_samples']}")
            print(f"   Features: {info['train_features']}")
            print(f"   Model saved at: {info.get('model_path', 'N/A')}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
