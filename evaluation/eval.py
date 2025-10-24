import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import pickle


def load_latest_model(model_type):
    """Load the most recent model of specified type (xgb, rf, lgbm, or catboost)"""
    models_dir = "models"

    if model_type == "lgbm":
        # LightGBM has a different naming convention
        pattern = "lgbm_model.pkl"
        model_files = glob.glob(os.path.join(models_dir, pattern))

        if not model_files:
            print(f"No {model_type} models found!")
            return None

        # Load LightGBM model
        latest_file = model_files[0]  # Only one file for now
        print(f"Loading {model_type} model: {os.path.basename(latest_file)}")

        # LightGBM model needs special loading
        import lightgbm as lgb

        lgb_model = lgb.Booster(model_file=latest_file)

        # Try to get feature names from another model for consistency
        feature_columns = None

        # Try to load XGBoost or RF model to get feature names
        for other_type in ["xgb", "rf"]:
            try:
                other_pattern = f"{other_type}_nfl_spread_model_*.pkl"
                other_files = glob.glob(os.path.join(models_dir, other_pattern))
                if other_files:
                    other_model = joblib.load(sorted(other_files)[-1])
                    feature_columns = other_model["feature_columns"]
                    break
            except:
                continue

        # If we couldn't get feature names from other models, create generic ones
        if feature_columns is None:
            feature_columns = [f"feature_{i}" for i in range(lgb_model.num_feature())]

        # Create a compatible data structure (we'll need to get feature names from other models)
        # For now, create a basic structure
        return {
            "model": lgb_model,
            "feature_columns": feature_columns,
            "accuracy": 0.5212,  # From your recent run
            "model_type": "lgbm",
        }

    elif model_type == "catboost":
        # CatBoost model saved as .pkl file but using CatBoost's save_model format
        pattern = "cat_boost_nfl_spread_model_*.pkl"
        model_files = glob.glob(os.path.join(models_dir, pattern))

        if not model_files:
            print(f"No {model_type} models found!")
            return None

        # Get the most recent model file
        latest_file = sorted(model_files)[-1]
        print(f"Loading {model_type} model: {os.path.basename(latest_file)}")

        # Load CatBoost model
        from catboost import CatBoostClassifier

        catboost_model = CatBoostClassifier()
        catboost_model.load_model(latest_file)

        # Load corresponding metadata file
        # Extract timestamp from filename (format: cat_boost_nfl_spread_model_YYYYMMDD_HHMMSS_accX.XXX.pkl)
        base_name = os.path.basename(latest_file)
        # Extract timestamp parts: split by _ and get parts 5 and 6, then join with _
        parts = base_name.split("_")
        timestamp = f"{parts[5]}_{parts[6]}"  # YYYYMMDD_HHMMSS
        metadata_pattern = f"cat_boost_metadata_{timestamp}.pkl"
        metadata_files = glob.glob(os.path.join(models_dir, metadata_pattern))

        if metadata_files:
            with open(metadata_files[0], "rb") as f:
                metadata = pickle.load(f)

            return {
                "model": catboost_model,
                "feature_columns": metadata["feature_columns"],
                "accuracy": metadata["accuracy"],
                "model_type": "catboost",
            }
        else:
            # Fallback if no metadata found
            print(
                f"Warning: No metadata found for CatBoost model. Looking for: {metadata_pattern}"
            )
            print(f"Available files: {os.listdir(models_dir)}")
            return {
                "model": catboost_model,
                "feature_columns": [
                    f"feature_{i}" for i in range(catboost_model.get_n_features_in())
                ],
                "accuracy": 0.0,
                "model_type": "catboost",
            }

    else:
        # XGBoost and Random Forest naming convention
        pattern = f"{model_type}_nfl_spread_model_*.pkl"
        model_files = glob.glob(os.path.join(models_dir, pattern))

        if not model_files:
            print(f"No {model_type} models found!")
            return None

        # Get the most recent file
        latest_file = sorted(model_files)[-1]
        print(f"Loading {model_type} model: {os.path.basename(latest_file)}")
        return joblib.load(latest_file)


def get_feature_importance(model_data, model_type, top_n=15):
    """Extract built-in feature importance from model"""
    model = model_data["model"]
    feature_columns = model_data["feature_columns"]

    if model_type == "xgb":
        # XGBoost feature importance
        importance = model.feature_importances_
    elif model_type == "rf":
        # Random Forest feature importance
        importance = model.feature_importances_
    elif model_type == "lgbm":
        # LightGBM feature importance
        importance = model.feature_importance(importance_type="gain")
    elif model_type == "catboost":
        # CatBoost feature importance
        importance = model.get_feature_importance()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create DataFrame with feature names and importance
    feature_df = (
        pd.DataFrame({"feature": feature_columns, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    return feature_df


def compare_feature_importance():
    """Compare feature importance between XGBoost, Random Forest, LightGBM, and CatBoost models"""

    # Load all four models
    xgb_data = load_latest_model("xgb")
    rf_data = load_latest_model("rf")
    lgbm_data = load_latest_model("lgbm")
    catboost_data = load_latest_model("catboost")

    # Check which models are available
    available_models = {}
    if xgb_data is not None:
        available_models["xgb"] = xgb_data
    if rf_data is not None:
        available_models["rf"] = rf_data
    if lgbm_data is not None:
        available_models["lgbm"] = lgbm_data
    if catboost_data is not None:
        available_models["catboost"] = catboost_data

    if len(available_models) == 0:
        print("Could not load any models!")
        return

    print("\nMODEL ACCURACY COMPARISON:")
    print("=" * 40)
    for model_type, model_data in available_models.items():
        model_name = {
            "xgb": "XGBoost",
            "rf": "Random Forest",
            "lgbm": "LightGBM",
            "catboost": "CatBoost",
        }[model_type]
        print(f"{model_name}: {model_data['accuracy']:.4f}")

    # Get feature importance for all available models
    print("\nExtracting feature importance...")
    model_features = {}
    for model_type, model_data in available_models.items():
        try:
            features = get_feature_importance(model_data, model_type, top_n=15)
            model_features[model_type] = features
        except Exception as e:
            print(f"Warning: Could not extract features for {model_type}: {e}")

    # Print top features for each model
    for model_type, features in model_features.items():
        model_name = {
            "xgb": "XGBOOST",
            "rf": "RANDOM FOREST",
            "lgbm": "LIGHTGBM",
            "catboost": "CATBOOST",
        }[model_type]
        print("\n" + "=" * 60)
        print(f"TOP 15 FEATURES - {model_name}")
        print("=" * 60)
        for i, row in features.iterrows():
            clean_name = row["feature"].replace("_", " ").title()
            print(f"{row.name + 1:2d}. {clean_name:<35} {row['importance']:.4f}")

    # Create comparison visualization
    num_models = len(model_features)
    if num_models > 0:
        fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 10))
        if num_models == 1:
            axes = [axes]

        colors = {
            "xgb": "steelblue",
            "rf": "forestgreen",
            "lgbm": "orange",
            "catboost": "purple",
        }
        model_names = {
            "xgb": "XGBoost",
            "rf": "Random Forest",
            "lgbm": "LightGBM",
            "catboost": "CatBoost",
        }

        for i, (model_type, features) in enumerate(model_features.items()):
            ax = axes[i]
            ax.barh(
                range(len(features)),
                features["importance"],
                color=colors[model_type],
                alpha=0.8,
            )
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(
                [f.replace("_", " ").title() for f in features["feature"]]
            )
            ax.set_xlabel("Feature Importance")
            ax.set_title(
                f'{model_names[model_type]} Feature Importance\nAccuracy: {available_models[model_type]["accuracy"]:.4f}'
            )
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("evaluation/model_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Find common top features across all models
        if len(model_features) >= 2:
            print("\n" + "=" * 60)
            print("MODEL FEATURE COMPARISON")
            print("=" * 60)

            # Get top 10 features for each model
            top_features = {}
            for model_type, features in model_features.items():
                top_features[model_type] = set(features.head(10)["feature"])

            # Find common features across all models
            if len(top_features) >= 2:
                common_features = set.intersection(*top_features.values())
                print(
                    f"\nCOMMON TOP 10 FEATURES (All {len(top_features)} Models Agree): {len(common_features)}"
                )
                for feature in sorted(common_features):
                    clean_name = feature.replace("_", " ").title()
                    print(f"• {clean_name}")

            # Show unique features for each model
            for model_type, features_set in top_features.items():
                other_features = set()
                for other_type, other_set in top_features.items():
                    if other_type != model_type:
                        other_features.update(other_set)

                unique_features = features_set - other_features
                model_name = model_names[model_type]
                print(f"\nUNIQUE TO {model_name.upper()}: {len(unique_features)}")
                for feature in sorted(unique_features):
                    clean_name = feature.replace("_", " ").title()
                    print(f"• {clean_name}")

    print(f"\nSaved comparison plot to: evaluation/model_comparison.png")


if __name__ == "__main__":
    compare_feature_importance()
