import nfl_data_py as nfl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime
import glob
import lightgbm as lgb


data_dir = "data"
data_files = glob.glob(os.path.join(data_dir, "processed_nfl_data.pkl"))

if not data_files:
    print("❌ No processed data file found!")
    print("   Looking for: data/processed_nfl_data.pkl")
    print(
        "   Run the data processing script first or use xgb_train.py to generate data"
    )
    exit(1)

print(f"Loading processed data from: {data_files[0]}")
data_files = joblib.load(data_files[0])

X_train = data_files["X_train"]
y_train = data_files["y_train"]
X_test = data_files["X_test"]
y_test = data_files["y_test"]

# Train LightGBM model
lgb_train = lgb.Dataset(X_train, label=y_train)
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 14,
    "verbose": -1,
}
lgb_model = lgb.train(params, lgb_train, num_boost_round=100)

# Evaluate model
y_pred = (lgb_model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Get and print feature importances
print("\n" + "=" * 60)
print("TOP FEATURES BY IMPORTANCE")
print("=" * 60)

feature_names = data_files["feature_columns"]
feature_importance = lgb_model.feature_importance(importance_type="gain")

# Create dataframe of feature importances
importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importance}
).sort_values("Importance", ascending=False)

# Print top 20 features
print("\nTop 20 Most Important Features:")
print("-" * 60)
for idx, row in importance_df.head(20).iterrows():
    print(f"{row['Feature']:50s} {row['Importance']:10.2f}")

print("\n" + "=" * 60)

# Save model
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "lgbm_model.pkl")
lgb_model.save_model(model_path)

print(f"Model saved to: {model_path}")
