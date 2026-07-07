# src/pipeline/preprocess.py (with parameter control)
import pandas as pd
from pathlib import Path
import yaml
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features import (
    GradientBoostingPreprocessor,
    LinearRegressionPreprocessor,
    RandomForestPreprocessor,
)


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()

    print("Loading raw data...")
    ratings_df = pd.read_csv("data/raw/ratings.csv")
    movies_df = pd.read_csv("data/raw/movies.csv")

    # Map preprocessor names to classes
    preprocessor_map = {
        "gradient_boosting": GradientBoostingPreprocessor,
        "linear_regression": LinearRegressionPreprocessor,
        "random_forest": RandomForestPreprocessor,
    }

    # Only run preprocessors that are enabled in params
    for name, enabled in params["preprocessors"].items():
        if enabled:
            print(f"Applying {name} preprocessor...")
            preprocessor = preprocessor_map[name]()
            processed_df = preprocessor.apply(ratings_df, movies_df)

            output_path = f"data/processed/{name}_data.csv"
            processed_df.to_csv(output_path, index=False)
            print(f"Saved {name} features: {processed_df.shape}")
        else:
            print(f"Skipping {name} preprocessor (disabled in params)")

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
