import joblib
import pandas as pd
import numpy as np
import os
import glob
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Try to import seaborn, use matplotlib if not available
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def load_all_models():
    """Load all 4 trained models: XGBoost, Random Forest, LightGBM, and CatBoost"""
    models_dir = "models"
    loaded_models = {}

    # Load XGBoost
    try:
        xgb_pattern = "xgb_nfl_spread_model_*.pkl"
        xgb_files = glob.glob(os.path.join(models_dir, xgb_pattern))
        if xgb_files:
            latest_xgb = sorted(xgb_files)[-1]
            xgb_data = joblib.load(latest_xgb)
            loaded_models["xgb"] = xgb_data
            print(f"✅ Loaded XGBoost: {os.path.basename(latest_xgb)}")
        else:
            print("❌ XGBoost model not found")
    except Exception as e:
        print(f"❌ Error loading XGBoost: {e}")

    # Load Random Forest
    try:
        rf_pattern = "rf_nfl_spread_model_*.pkl"
        rf_files = glob.glob(os.path.join(models_dir, rf_pattern))
        if rf_files:
            latest_rf = sorted(rf_files)[-1]
            rf_data = joblib.load(latest_rf)
            loaded_models["rf"] = rf_data
            print(f"✅ Loaded Random Forest: {os.path.basename(latest_rf)}")
        else:
            print("❌ Random Forest model not found")
    except Exception as e:
        print(f"❌ Error loading Random Forest: {e}")

    # Load LightGBM
    try:
        lgbm_pattern = "lgbm_model.pkl"
        lgbm_files = glob.glob(os.path.join(models_dir, lgbm_pattern))
        if lgbm_files:
            import lightgbm as lgb

            lgb_model = lgb.Booster(model_file=lgbm_files[0])

            # Get feature names from another model for consistency
            feature_columns = None
            if "xgb" in loaded_models:
                feature_columns = loaded_models["xgb"]["feature_columns"]
            elif "rf" in loaded_models:
                feature_columns = loaded_models["rf"]["feature_columns"]

            if feature_columns is None:
                feature_columns = [
                    f"feature_{i}" for i in range(lgb_model.num_feature())
                ]

            loaded_models["lgbm"] = {
                "model": lgb_model,
                "feature_columns": feature_columns,
                "model_type": "lgbm",
            }
            print(f"✅ Loaded LightGBM: {os.path.basename(lgbm_files[0])}")
        else:
            print("❌ LightGBM model not found")
    except Exception as e:
        print(f"❌ Error loading LightGBM: {e}")

    # Load CatBoost
    try:
        catboost_pattern = "cat_boost_nfl_spread_model_*.pkl"
        catboost_files = glob.glob(os.path.join(models_dir, catboost_pattern))
        if catboost_files:
            from catboost import CatBoostClassifier

            latest_catboost = sorted(catboost_files)[-1]

            # Load CatBoost model
            catboost_model = CatBoostClassifier()
            catboost_model.load_model(latest_catboost)

            # Load metadata
            base_name = os.path.basename(latest_catboost)
            parts = base_name.split("_")
            timestamp = f"{parts[5]}_{parts[6]}"
            metadata_pattern = f"cat_boost_metadata_{timestamp}.pkl"
            metadata_files = glob.glob(os.path.join(models_dir, metadata_pattern))

            if metadata_files:
                with open(metadata_files[0], "rb") as f:
                    metadata = pickle.load(f)

                loaded_models["catboost"] = {
                    "model": catboost_model,
                    "feature_columns": metadata["feature_columns"],
                    "model_type": "catboost",
                }
                print(f"✅ Loaded CatBoost: {os.path.basename(latest_catboost)}")
            else:
                print("❌ CatBoost metadata not found")
        else:
            print("❌ CatBoost model not found")
    except Exception as e:
        print(f"❌ Error loading CatBoost: {e}")

    return loaded_models


def load_test_data():
    """Load the processed test data (2022-2023)"""
    try:
        test_data_path = "data/processed_nfl_data.pkl"
        if not os.path.exists(test_data_path):
            print(
                "❌ Processed data not found. Please run the data download script first."
            )
            return None, None, None, None

        print("Loading processed test data...")

        # Load the data using joblib (as saved by download_data.py)
        data = joblib.load(test_data_path)

        # Extract test data components
        X_test = data["X_test"]
        y_test = data["y_test"]
        feature_columns = data["feature_columns"]
        label_encoders = data["label_encoders"]

        print(f"✅ Test data loaded successfully:")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {len(feature_columns)}")
        print(f"   Target distribution: {y_test.value_counts().to_dict()}")

        return X_test, y_test, label_encoders, feature_columns

    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        print(
            "Make sure to run 'python scripts/data_scripts/download_data.py' first to generate processed data."
        )
        return None, None, None, None


def make_ensemble_predictions(models, X_test, y_test, label_encoders, feature_columns):
    """Make predictions using all models and return averaged probabilities"""
    predictions = {}
    probabilities = {}

    print("\nMaking predictions with each model...")

    for model_name, model_data in models.items():
        try:
            model = model_data["model"]
            model_features = model_data["feature_columns"]

            # Special handling for CatBoost - decode categorical features FIRST
            if model_name == "catboost":
                print("🔄 Preparing categorical features for CatBoost...")
                print(f"   🔍 Model data keys: {list(model_data.keys())}")

                X_aligned = X_test.copy()

                # Get categorical features that need decoding
                categorical_features = model_data.get("categorical_features", [])
                print(f"   📋 Categorical features from model: {categorical_features}")

                if not categorical_features:
                    print(
                        "   ⚠️  No categorical features found! Trying to get from metadata..."
                    )
                    # Use the variables from the outer scope since we're in the CatBoost loading section
                    categorical_features = [
                        "away_team",
                        "home_team",
                        "div_game",
                        "away_qb_name",
                        "home_qb_name",
                        "away_coach",
                        "home_coach",
                        "referee",
                    ]
                    print(
                        f"   🔧 Using hardcoded categorical features: {categorical_features}"
                    )

                # Map encoded column names to original names for CatBoost
                encoded_to_original = {
                    "away_team_encoded": "away_team",
                    "home_team_encoded": "home_team",
                    "div_game_encoded": "div_game",
                    "away_qb_name_encoded": "away_qb_name",
                    "home_qb_name_encoded": "home_qb_name",
                    "away_coach_encoded": "away_coach",
                    "home_coach_encoded": "home_coach",
                    "referee_encoded": "referee",
                }

                for encoded_col, original_col in encoded_to_original.items():
                    if (
                        encoded_col in X_aligned.columns
                        and original_col in categorical_features
                    ):
                        if original_col in label_encoders:
                            # Decode from numbers back to original strings
                            le = label_encoders[original_col]
                            try:
                                decoded_values = le.inverse_transform(
                                    X_aligned[encoded_col]
                                )
                                X_aligned[original_col] = decoded_values
                                print(f"   ✅ Decoded {encoded_col} -> {original_col}")
                            except Exception as e:
                                print(f"   ❌ Could not decode {encoded_col}: {e}")
                                continue
                        else:
                            print(f"   ⚠️  No label encoder found for {original_col}")
                    else:
                        if encoded_col not in X_aligned.columns:
                            print(f"   ⚠️  Column {encoded_col} not found in data")
                        if original_col not in categorical_features:
                            print(
                                f"   ⚠️  {original_col} not in categorical features list"
                            )

                print(
                    f"   📋 Available columns after decoding: {list(X_aligned.columns)}"
                )
                print(f"   📋 Categorical features expected: {categorical_features}")

                # Remove encoded columns and keep only the original ones that CatBoost expects
                cols_to_keep = []
                for col in X_aligned.columns:
                    if col in categorical_features:
                        cols_to_keep.append(col)
                    elif col not in encoded_to_original.keys():
                        cols_to_keep.append(col)

                X_aligned = X_aligned[cols_to_keep]
                print(
                    f"   📊 Final feature count for CatBoost: {len(X_aligned.columns)}"
                )

                # Now align with model features
                if set(model_features) != set(X_aligned.columns):
                    print(f"⚠️  Feature mismatch for {model_name}, aligning features...")
                    X_aligned = X_aligned[model_features]

            else:
                # Standard feature alignment for other models
                if set(model_features) != set(feature_columns):
                    print(f"⚠️  Feature mismatch for {model_name}, aligning features...")
                    X_aligned = X_test[model_features]
                else:
                    X_aligned = X_test[feature_columns]

            if model_name == "lgbm":
                # LightGBM uses different prediction method
                pred_proba = model.predict(
                    X_aligned, num_iteration=model.best_iteration
                )
                # Convert to binary probabilities
                pred_proba_binary = np.column_stack([1 - pred_proba, pred_proba])
                pred_class = (pred_proba > 0.5).astype(int)
            else:
                # Standard sklearn interface
                pred_proba_binary = model.predict_proba(X_aligned)
                pred_class = model.predict(X_aligned)

            predictions[model_name] = pred_class
            probabilities[model_name] = pred_proba_binary[
                :, 1
            ]  # Probability of class 1

            accuracy = accuracy_score(y_test, pred_class)
            print(f"✅ {model_name.upper()}: {accuracy:.4f}")

        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            continue

    return predictions, probabilities


def ensemble_average(probabilities):
    """Calculate ensemble predictions by averaging probabilities"""
    if not probabilities:
        return None, None

    # Stack all probabilities and calculate mean
    prob_stack = np.column_stack(list(probabilities.values()))
    ensemble_prob = np.mean(prob_stack, axis=1)
    ensemble_pred = (ensemble_prob > 0.5).astype(int)

    return ensemble_pred, ensemble_prob


def evaluate_ensemble_performance(
    models, X_test, y_test, label_encoders, feature_columns
):
    """Evaluate individual models and ensemble performance"""
    print("=" * 80)
    print("ENSEMBLE MODEL EVALUATION")
    print("=" * 80)

    # Make predictions with all models
    predictions, probabilities = make_ensemble_predictions(
        models, X_test, y_test, label_encoders, feature_columns
    )

    if not probabilities:
        print("❌ No valid predictions made. Cannot create ensemble.")
        return

    # Calculate ensemble predictions
    ensemble_pred, ensemble_prob = ensemble_average(probabilities)

    # Evaluate individual models
    print(f"\n{'Model':<15} {'Accuracy':<10} {'Available':<10}")
    print("-" * 40)

    individual_accuracies = {}
    for model_name in ["xgb", "rf", "lgbm", "catboost"]:
        if model_name in predictions:
            acc = accuracy_score(y_test, predictions[model_name])
            individual_accuracies[model_name] = acc
            print(f"{model_name.upper():<15} {acc:.4f}     ✅")
        else:
            print(f"{model_name.upper():<15} {'N/A':<10}     ❌")

    # Evaluate ensemble
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    print(f"{'ENSEMBLE':<15} {ensemble_accuracy:.4f}     🚀")

    # Show improvement
    if individual_accuracies:
        best_individual = max(individual_accuracies.values())
        improvement = ensemble_accuracy - best_individual
        print(f"\n📊 ENSEMBLE PERFORMANCE:")
        print(f"   Best Individual Model: {best_individual:.4f}")
        print(f"   Ensemble Accuracy:     {ensemble_accuracy:.4f}")
        print(f"   Improvement:           {improvement:+.4f}")
        print(f"   Models Used:           {len(probabilities)}")

    # Detailed classification report
    print(f"\n📋 DETAILED ENSEMBLE RESULTS:")
    print(
        classification_report(
            y_test, ensemble_pred, target_names=["Away Win", "Home Win"]
        )
    )

    # Create confusion matrix visualization
    plt.figure(figsize=(12, 5))

    # Individual model accuracies
    plt.subplot(1, 2, 1)
    model_names = list(individual_accuracies.keys()) + ["ENSEMBLE"]
    accuracies = list(individual_accuracies.values()) + [ensemble_accuracy]
    colors = ["steelblue", "forestgreen", "orange", "purple", "red"][: len(accuracies)]

    bars = plt.bar(model_names, accuracies, color=colors, alpha=0.7)
    plt.ylabel("Accuracy")
    plt.title("Model Performance Comparison")
    plt.ylim(0.45, 0.60)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=45)

    # Confusion matrix
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, ensemble_pred)

    if HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Away Win", "Home Win"],
            yticklabels=["Away Win", "Home Win"],
        )
    else:
        # Use matplotlib if seaborn is not available
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.colorbar()

        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")

        plt.xticks([0, 1], ["Away Win", "Home Win"])
        plt.yticks([0, 1], ["Away Win", "Home Win"])

    plt.title("Ensemble Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig("evaluation/ensemble_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n💾 Results saved to: evaluation/ensemble_results.png")

    return ensemble_pred, ensemble_prob, individual_accuracies


def main():
    """Main function to run ensemble evaluation"""
    print("🚀 NFL SPREAD PREDICTION - ENSEMBLE MODEL EVALUATION")
    print("=" * 60)

    # Load all trained models
    print("Loading trained models...")
    models = load_all_models()

    if not models:
        print("❌ No models found! Please train some models first.")
        return

    print(f"\n✅ Successfully loaded {len(models)} models: {list(models.keys())}")

    # Load test data
    print("\nLoading test data...")
    X_test, y_test, label_encoders, feature_columns = load_test_data()

    if X_test is None:
        print(
            "❌ Could not load test data! Please run 'python scripts/data_scripts/download_data.py' first."
        )
        return

    print(
        f"✅ Test data loaded: {len(X_test)} samples, {len(feature_columns)} features"
    )

    # Evaluate ensemble performance
    ensemble_pred, ensemble_prob, individual_accuracies = evaluate_ensemble_performance(
        models, X_test, y_test, label_encoders, feature_columns
    )

    print("\n🎯 ENSEMBLE EVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
