import pandas as pd
from pathlib import Path
import yaml
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features import (
    GradientBoostingFeatureEng,
    LinearRegressionFeatureEng,
    RandomForestFeatureEng,
    MLPFeatureEngineer,
)


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()

    feature_map = {
        "gradient_boosting": GradientBoostingFeatureEng,
        "linear_regression": LinearRegressionFeatureEng,
        "random_forest": RandomForestFeatureEng,
        "multi_layer_perceptron": MLPFeatureEngineer,
    }

    for name, enabled in params["features_eng"].items():
        if enabled:
            print(f"\nApplying {name} feature engineering...")

            # Load pre-processed data
            df = pd.read_csv(f"data/processed/{name}_data.csv")

            # Fit and transform
            features_eng = feature_map[name]().fit(df)
            X, y = features_eng.transform(df)

            # Combine features and target
            processed_df = pd.DataFrame(
                X,
                columns=features_eng.encoders.get(
                    "feature_columns", [f"f{i}" for i in range(X.shape[1])]
                ),
            )
            processed_df["rating"] = y.values

            # Save
            output_path = f"data/features/{name}_features.csv"
            processed_df.to_csv(output_path, index=False)
            print(f"Saved {name} features: {processed_df.shape}")
        else:
            print(f"Skipping {name} feature engineering (disabled in params)")

    print("\nFeature engineering complete!")


if __name__ == "__main__":
    main()
