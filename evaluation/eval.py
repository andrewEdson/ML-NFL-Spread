import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def load_latest_model(model_type):
    """Load the most recent model of specified type (xgb or rf)"""
    models_dir = "models"
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
    """Compare feature importance between XGBoost and Random Forest models"""

    # Load both models
    xgb_data = load_latest_model("xgb")
    rf_data = load_latest_model("rf")

    if xgb_data is None or rf_data is None:
        print("Could not load both models!")
        return

    print(f"\nXGBoost Accuracy: {xgb_data['accuracy']:.4f}")
    print(f"Random Forest Accuracy: {rf_data['accuracy']:.4f}")

    # Get feature importance for both models
    print("\nExtracting feature importance...")
    xgb_features = get_feature_importance(xgb_data, "xgb", top_n=15)
    rf_features = get_feature_importance(rf_data, "rf", top_n=15)

    # Print top features
    print("\n" + "=" * 60)
    print("TOP 15 FEATURES - XGBOOST")
    print("=" * 60)
    for i, row in xgb_features.iterrows():
        clean_name = row["feature"].replace("_", " ").title()
        print(f"{row.name + 1:2d}. {clean_name:<35} {row['importance']:.4f}")

    print("\n" + "=" * 60)
    print("TOP 15 FEATURES - RANDOM FOREST")
    print("=" * 60)
    for i, row in rf_features.iterrows():
        clean_name = row["feature"].replace("_", " ").title()
        print(f"{row.name + 1:2d}. {clean_name:<35} {row['importance']:.4f}")

    # Create comparison visualization
    plt.figure(figsize=(15, 10))

    # XGBoost subplot
    plt.subplot(1, 2, 1)
    plt.barh(
        range(len(xgb_features)),
        xgb_features["importance"],
        color="steelblue",
        alpha=0.8,
    )
    plt.yticks(
        range(len(xgb_features)),
        [f.replace("_", " ").title() for f in xgb_features["feature"]],
    )
    plt.xlabel("Feature Importance")
    plt.title(f'XGBoost Feature Importance\nAccuracy: {xgb_data["accuracy"]:.4f}')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)

    # Random Forest subplot
    plt.subplot(1, 2, 2)
    plt.barh(
        range(len(rf_features)),
        rf_features["importance"],
        color="forestgreen",
        alpha=0.8,
    )
    plt.yticks(
        range(len(rf_features)),
        [f.replace("_", " ").title() for f in rf_features["feature"]],
    )
    plt.xlabel("Feature Importance")
    plt.title(f'Random Forest Feature Importance\nAccuracy: {rf_data["accuracy"]:.4f}')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evaluation/model_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Find common top features
    xgb_top10 = set(xgb_features.head(10)["feature"])
    rf_top10 = set(rf_features.head(10)["feature"])
    common_features = xgb_top10.intersection(rf_top10)

    print("\n" + "=" * 60)
    print("COMMON TOP 10 FEATURES (Both Models Agree)")
    print("=" * 60)
    for feature in common_features:
        clean_name = feature.replace("_", " ").title()
        print(f"• {clean_name}")

    print(f"\nSaved comparison plot to: evaluation/model_comparison.png")


if __name__ == "__main__":
    compare_feature_importance()
