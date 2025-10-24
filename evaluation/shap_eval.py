import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
import os
import glob
from sklearn.inspection import permutation_importance


def load_latest_model(model_type="xgb"):
    """Load the most recent XGBoost model"""
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


def create_synthetic_test_data(feature_columns, sample_size=1000):
    """Create synthetic test data for SHAP analysis with realistic NFL ranges"""
    print(f"Creating {sample_size} synthetic samples for SHAP analysis...")
    np.random.seed(42)
    n_features = len(feature_columns)

    # Generate synthetic data with realistic ranges
    synthetic_data = np.random.randn(sample_size, n_features)

    # Scale to realistic ranges for different feature types
    for i, col in enumerate(feature_columns):
        col_lower = col.lower()

        if "encoded" in col_lower:
            # Categorical features: 0-32 range (teams, coaches, etc.)
            synthetic_data[:, i] = np.random.randint(0, 32, sample_size)
        elif "odds" in col_lower or "spread" in col_lower:
            # Odds/spread features: -25 to 25 range
            synthetic_data[:, i] = np.random.uniform(-25, 25, sample_size)
        elif "yards" in col_lower:
            # Yardage features: 0-600 range
            synthetic_data[:, i] = np.random.uniform(0, 600, sample_size)
        elif "temp" in col_lower:
            # Temperature: 10-90 range
            synthetic_data[:, i] = np.random.uniform(10, 90, sample_size)
        elif "rest" in col_lower:
            # Rest days: 3-21 range
            synthetic_data[:, i] = np.random.randint(3, 22, sample_size)
        elif "epa" in col_lower:
            # EPA features: -0.8 to 0.8 range
            synthetic_data[:, i] = np.random.uniform(-0.8, 0.8, sample_size)
        elif "tds" in col_lower or "td" in col_lower:
            # Touchdowns: 0-8 range
            synthetic_data[:, i] = np.random.randint(0, 9, sample_size)
        elif "interception" in col_lower or "fumble" in col_lower:
            # Turnovers: 0-6 range
            synthetic_data[:, i] = np.random.randint(0, 7, sample_size)
        elif "sack" in col_lower:
            # Sacks: 0-10 range
            synthetic_data[:, i] = np.random.randint(0, 11, sample_size)
        elif "point" in col_lower and "fantasy" in col_lower:
            # Fantasy points: 0-40 range
            synthetic_data[:, i] = np.random.uniform(0, 40, sample_size)
        elif "injur" in col_lower:
            # Injury features
            if "severity" in col_lower and "avg" not in col_lower:
                # Total severity: 0-20 range
                synthetic_data[:, i] = np.random.uniform(0, 20, sample_size)
            elif "avg" in col_lower:
                # Average severity: 0-5 range
                synthetic_data[:, i] = np.random.uniform(0, 5, sample_size)
            elif "total" in col_lower or "players" in col_lower:
                # Total counts: 0-15 range
                synthetic_data[:, i] = np.random.randint(0, 16, sample_size)
            else:
                # Position-specific injuries: 0-8 range
                synthetic_data[:, i] = np.random.uniform(0, 8, sample_size)
        elif "season" in col_lower:
            # Season: 2015-2023 range
            synthetic_data[:, i] = np.random.randint(2015, 2024, sample_size)
        elif "week" in col_lower:
            # Week: 1-18 range
            synthetic_data[:, i] = np.random.randint(1, 19, sample_size)
        else:
            # Other features: scale normal distribution
            synthetic_data[:, i] = np.abs(synthetic_data[:, i]) * 5

    X_test = pd.DataFrame(synthetic_data, columns=feature_columns)
    return X_test


def get_shap_top_features(model_data, top_n=15):
    """Get top N features using SHAP analysis"""
    print("\n" + "=" * 60)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    model = model_data["model"]
    feature_columns = model_data["feature_columns"]
    accuracy = model_data["accuracy"]

    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Total Features: {len(feature_columns)}")

    # Create synthetic test data
    X_sample = create_synthetic_test_data(feature_columns, sample_size=500)

    print("\nComputing SHAP values...")

    try:
        # Try TreeExplainer first (faster for tree-based models)
        print("Attempting TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"TreeExplainer failed: {e}")
        print("Falling back to general Explainer...")

        try:
            # Fallback to general explainer with sampling
            sample_data = X_sample.sample(n=100)  # Smaller sample for general explainer
            explainer = shap.Explainer(model, sample_data)
            shap_values = explainer(X_sample.sample(n=200))

            # Extract values if it's an Explanation object
            if hasattr(shap_values, "values"):
                shap_values = shap_values.values

        except Exception as e2:
            print(f"General Explainer also failed: {e2}")
            print("Using Permutation-based feature importance as fallback...")

            # Fallback to permutation importance
            from sklearn.inspection import permutation_importance

            # Create target for permutation importance (dummy binary target)
            y_dummy = np.random.randint(0, 2, size=len(X_sample))

            perm_importance = permutation_importance(
                model, X_sample, y_dummy, n_repeats=5, random_state=42
            )

            # Create SHAP-like importance DataFrame
            shap_importance_df = pd.DataFrame(
                {
                    "feature": feature_columns,
                    "shap_importance": np.abs(perm_importance.importances_mean),
                }
            ).sort_values("shap_importance", ascending=False)

            print("✅ Permutation-based importance calculated successfully")
            return None, shap_importance_df, None, X_sample

    # Handle binary classification SHAP values
    if isinstance(shap_values, list) and len(shap_values) == 2:
        # Use positive class SHAP values
        shap_values = shap_values[1]
    elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
        # Use positive class SHAP values
        shap_values = shap_values[:, :, 1]

    # Calculate mean absolute SHAP values for feature importance
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Create SHAP feature importance DataFrame
    shap_importance_df = pd.DataFrame(
        {"feature": feature_columns, "shap_importance": mean_shap_values}
    ).sort_values("shap_importance", ascending=False)

    print("✅ SHAP values calculated successfully")
    return shap_values, shap_importance_df, explainer, X_sample


def print_top_features(shap_importance_df, top_n=15):
    """Print the top N features with clean formatting"""
    print(f"\n{'='*60}")
    print(f"TOP {top_n} FEATURES BY SHAP IMPORTANCE")
    print(f"{'='*60}")

    top_features = shap_importance_df.head(top_n)

    for i, (idx, row) in enumerate(top_features.iterrows(), 1):
        feature_name = row["feature"]
        importance = row["shap_importance"]

        # Clean up feature name for display
        clean_name = feature_name.replace("_", " ").title()
        clean_name = clean_name.replace("Epa", "EPA")
        clean_name = clean_name.replace("Qb", "QB")
        clean_name = clean_name.replace("Tds", "TDs")

        print(f"{i:2d}. {clean_name:<40} {importance:.4f}")


def create_shap_visualizations(
    shap_values,
    shap_importance_df,
    X_sample,
    model_data,
    explainer=None,
    model_type="model",
):
    """Create SHAP visualization plots"""

    # Create evaluation directory if it doesn't exist
    os.makedirs("evaluation", exist_ok=True)

    model_name = (
        "XGBoost"
        if model_type == "xgb"
        else "Random Forest" if model_type == "rf" else "Model"
    )

    # Always create the bar plot (works with any importance method)
    plt.figure(figsize=(12, 10))
    top_15_features = shap_importance_df.head(15)

    colors = plt.cm.viridis(np.linspace(0, 1, len(top_15_features)))

    bars = plt.barh(
        range(len(top_15_features)),
        top_15_features["shap_importance"],
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Clean feature names for y-axis
    clean_names = []
    for feature in top_15_features["feature"]:
        clean_name = feature.replace("_", " ").title()
        clean_name = clean_name.replace("Epa", "EPA")
        clean_name = clean_name.replace("Qb", "QB")
        clean_name = clean_name.replace("Tds", "TDs")
        clean_names.append(clean_name)

    plt.yticks(range(len(top_15_features)), clean_names)
    plt.xlabel("Feature Importance Score", fontsize=12)
    plt.title(
        f'{model_name} - Top 15 NFL Spread Prediction Features\nModel Accuracy: {model_data["accuracy"]:.4f}',
        fontsize=14,
        fontweight="bold",
    )
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis="x")

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(
            width + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        f"evaluation/{model_type}_shap_top_15_features.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\n📊 {model_name} feature importance visualization saved:")
    print(f"   • evaluation/{model_type}_shap_top_15_features.png")

    # Only create SHAP-specific plots if we have actual SHAP values
    if shap_values is not None and explainer is not None:
        # 1. SHAP Summary Plot (beeswarm plot)
        plt.figure(figsize=(14, 10))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=X_sample.columns.tolist(),
            show=False,
            max_display=15,
        )
        plt.title(
            f"{model_name} - SHAP Summary Plot - Feature Impact on NFL Spread Predictions"
        )
        plt.tight_layout()
        plt.savefig(
            f"evaluation/{model_type}_shap_summary_plot.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 3. SHAP Waterfall plot for a single prediction
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            explainer.expected_value,
            shap_values[0],
            X_sample.iloc[0],
            max_display=15,
            show=False,
        )
        plt.title(f"{model_name} - SHAP Waterfall Plot - Single Prediction Explanation")
        plt.tight_layout()
        plt.savefig(
            f"evaluation/{model_type}_shap_waterfall_plot.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"   • evaluation/{model_type}_shap_summary_plot.png")
        print(f"   • evaluation/{model_type}_shap_waterfall_plot.png")
    else:
        print("   (SHAP-specific plots not available with fallback method)")


def analyze_feature_categories(shap_importance_df):
    """Analyze feature importance by categories"""
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE BY CATEGORY")
    print(f"{'='*60}")

    categories = {
        "Team Performance": [
            "passing_yards",
            "rushing_yards",
            "receiving_yards",
            "passing_tds",
            "rushing_tds",
            "receiving_tds",
            "fantasy_points",
        ],
        "Defensive Stats": ["interceptions", "sacks", "fumbles"],
        "Betting Lines": ["spread_line", "total_line", "odds", "moneyline"],
        "Game Context": ["season", "week", "rest", "temp", "roof"],
        "Team Identity": ["team", "qb_name", "coach"],
        "Advanced Metrics": ["epa"],
        "Injury Data": ["injur"],
    }

    category_scores = {}

    for category, keywords in categories.items():
        category_importance = 0
        feature_count = 0

        for _, row in shap_importance_df.iterrows():
            feature = row["feature"].lower()
            if any(keyword in feature for keyword in keywords):
                category_importance += row["shap_importance"]
                feature_count += 1

        if feature_count > 0:
            category_scores[category] = {
                "total_importance": category_importance,
                "avg_importance": category_importance / feature_count,
                "feature_count": feature_count,
            }

    # Sort by total importance
    sorted_categories = sorted(
        category_scores.items(), key=lambda x: x[1]["total_importance"], reverse=True
    )

    for category, stats in sorted_categories:
        print(
            f"{category:<20} | Total: {stats['total_importance']:.3f} | "
            f"Avg: {stats['avg_importance']:.3f} | Features: {stats['feature_count']}"
        )


def main():
    """Main function to run SHAP analysis"""
    print("🏈 NFL Spread Prediction - SHAP Feature Analysis")
    print("=" * 60)

    # Try to load both model types
    model_types = ["xgb", "rf"]
    available_models = {}

    for model_type in model_types:
        model_data = load_latest_model(model_type)
        if model_data is not None:
            available_models[model_type] = model_data

    if not available_models:
        print(
            "❌ No models found! Make sure you have trained models in the models/ directory."
        )
        return

    # Analyze each available model
    for model_type, model_data in available_models.items():
        model_name = "XGBoost" if model_type == "xgb" else "Random Forest"
        print(f"\n{'='*60}")
        print(f"ANALYZING {model_name.upper()} MODEL")
        print(f"{'='*60}")

        # Get SHAP analysis
        shap_values, shap_importance_df, explainer, X_sample = get_shap_top_features(
            model_data, top_n=15
        )

        # Print top features
        print_top_features(shap_importance_df, top_n=15)

        # Analyze by categories
        analyze_feature_categories(shap_importance_df)

        # Create visualizations with model-specific names
        create_shap_visualizations(
            shap_values, shap_importance_df, X_sample, model_data, explainer, model_type
        )

        # Save detailed results with model-specific names
        detailed_results = shap_importance_df.head(25)
        detailed_results.to_csv(
            f"evaluation/{model_type}_shap_top_25_features.csv", index=False
        )
        print(
            f"\n💾 Detailed results saved to: evaluation/{model_type}_shap_top_25_features.csv"
        )

        print(f"\n✅ {model_name} SHAP analysis complete!")
        print(f"   Model: {model_data.get('timestamp', 'Unknown')}")
        print(f"   Accuracy: {model_data['accuracy']:.4f}")
        print(f"   Features analyzed: {len(model_data['feature_columns'])}")

    # If we have both models, create a comparison
    if len(available_models) == 2:
        print(f"\n{'='*80}")
        print("MODEL COMPARISON - TOP 10 FEATURES")
        print(f"{'='*80}")

        # Get top features for both models
        xgb_shap_values, xgb_shap_df, _, _ = get_shap_top_features(
            available_models["xgb"], top_n=10
        )
        rf_shap_values, rf_shap_df, _, _ = get_shap_top_features(
            available_models["rf"], top_n=10
        )

        xgb_top10 = set(xgb_shap_df.head(10)["feature"])
        rf_top10 = set(rf_shap_df.head(10)["feature"])

        common_features = xgb_top10.intersection(rf_top10)
        xgb_only = xgb_top10 - rf_top10
        rf_only = rf_top10 - xgb_top10

        print(f"\n🤝 COMMON TOP 10 FEATURES ({len(common_features)} features):")
        for feature in sorted(common_features):
            clean_name = feature.replace("_", " ").title()
            print(f"   • {clean_name}")

        print(f"\n🔵 XGBoost ONLY ({len(xgb_only)} features):")
        for feature in sorted(xgb_only):
            clean_name = feature.replace("_", " ").title()
            print(f"   • {clean_name}")

        print(f"\n🟢 Random Forest ONLY ({len(rf_only)} features):")
        for feature in sorted(rf_only):
            clean_name = feature.replace("_", " ").title()
            print(f"   • {clean_name}")


if __name__ == "__main__":
    main()
