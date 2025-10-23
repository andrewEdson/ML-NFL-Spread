import nfl_data_py as nfl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime

# Get game schedules with scores and spreads (perfect for spread prediction!)
years = list(range(1999, 2024))  # Further reduced range for faster loading and testing
schedules = nfl.import_schedules(years)

# Import player weekly stats and aggregate to team level
player_stats = nfl.import_weekly_data(years)

# Create team-level aggregations from player stats
team_stats = (
    player_stats.groupby(["recent_team", "season", "week"])
    .agg(
        {
            "passing_yards": "sum",
            "passing_tds": "sum",
            "interceptions": "sum",
            "sacks": "sum",
            "rushing_yards": "sum",
            "rushing_tds": "sum",
            "receiving_yards": "sum",
            "receiving_tds": "sum",
            "fantasy_points": "sum",
        }
    )
    .reset_index()
)

# Rename the team column for easier merging
team_stats = team_stats.rename(columns={"recent_team": "team"})

# Merge team stats into schedules for both home and away teams
schedules = schedules.merge(
    team_stats,
    left_on=["season", "week", "home_team"],
    right_on=["season", "week", "team"],
    suffixes=("", "_home"),
    how="left",
).drop(columns=["team"], errors="ignore")

# Merge team stats for away team
schedules = schedules.merge(
    team_stats,
    left_on=["season", "week", "away_team"],
    right_on=["season", "week", "team"],
    suffixes=("", "_away"),
    how="left",
).drop(columns=["team"], errors="ignore")

# Filter for regular season games only and remove games without spread data
df = schedules[
    (schedules["game_type"] == "REG")
    & (schedules["spread_line"].notna())
    & (schedules["away_score"].notna())
].copy()

# Create key features for spread prediction
df["actual_spread"] = (
    df["home_score"] - df["away_score"]
)  # Positive = home team won by more
df["spread_diff"] = (
    df["actual_spread"] - df["spread_line"]
)  # How much the actual beat the spread
df["home_covered"] = df["spread_diff"] > 0  # Did home team cover the spread?

team_stat_cols = [
    col
    for col in df.columns
    if any(
        stat in col
        for stat in [
            "passing_yards",
            "passing_tds",
            "interceptions",
            "sacks",
            "rushing_yards",
            "rushing_tds",
            "receiving_yards",
            "receiving_tds",
            "fantasy_points",
        ]
    )
]

# Prepare features and target variable
from sklearn.preprocessing import LabelEncoder

# First, let's select only numerical columns and a few key categorical ones
base_numerical_features = [
    "season",
    "week",
    "away_rest",
    "home_rest",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
    "temp",
    "wind",
    "pff",
]

# Add team stats dynamically based on what's actually available
team_stat_base_names = [
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "fantasy_points",
]

team_stats_features = []
for stat in team_stat_base_names:
    # Check for home team stats (no suffix means home team from first merge)
    if stat in df.columns:
        team_stats_features.append(stat)
    # Check for away team stats
    if f"{stat}_away" in df.columns:
        team_stats_features.append(f"{stat}_away")

numerical_features = base_numerical_features + team_stats_features

categorical_features = [
    "away_team",
    "home_team",
    "roof",
    "surface",
    "div_game",
    "away_qb_name",
    "home_qb_name",
    "away_coach",
    "home_coach",
    "referee",
]

# Create a clean dataset with only these features
df_clean = df[numerical_features + categorical_features + ["home_covered"]].copy()

# Debug: Check data types and missing values
print("\nDebugging data types:")
for col in numerical_features:
    if col in df_clean.columns:
        print(f"{col}: {df_clean[col].dtype}, missing: {df_clean[col].isna().sum()}")
        # Convert to numeric, coercing errors to NaN
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

print("\nDebugging categorical features:")
for col in categorical_features:
    if col in df_clean.columns:
        print(f"{col}: {df_clean[col].dtype}, missing: {df_clean[col].isna().sum()}")

# Fill missing values instead of dropping entire games
print("\nFilling missing values...")

# Fill numerical missing values with median/0
for col in numerical_features:
    if col in df_clean.columns and df_clean[col].isna().any():
        if col in ["temp", "wind", "pff"]:  # Weather/ratings - use median
            fill_value = df_clean[col].median()
        else:  # Team stats - use 0 (means team didn't record that stat)
            fill_value = 0
        df_clean[col] = df_clean[col].fillna(fill_value)
        print(f"Filled {col} missing values with {fill_value}")

# Fill categorical missing values with 'Unknown'
for col in categorical_features:
    if col in df_clean.columns and df_clean[col].isna().any():
        df_clean[col] = df_clean[col].fillna("Unknown")
        print(f"Filled {col} missing values with 'Unknown'")

# Only drop games missing absolutely critical data (spread line, target variable)
critical_cols = ["spread_line", "home_covered"]
before_critical_drop = len(df_clean)
df_clean = df_clean.dropna(subset=critical_cols)
after_critical_drop = len(df_clean)
print(
    f"Dropped {before_critical_drop - after_critical_drop} games missing critical data (spread/result)"
)
print(f"Final games retained: {len(df_clean)} (vs original {len(df)} games)")

# Encode categorical variables using Label Encoding
le_dict = {}
for col in categorical_features:
    if col in df_clean.columns:
        le = LabelEncoder()
        df_clean[col + "_encoded"] = le.fit_transform(df_clean[col].astype(str))
        le_dict[col] = le  # Save encoder for later use

# Select features (numerical + encoded categorical)
available_numerical = [col for col in numerical_features if col in df_clean.columns]
available_categorical = [
    col + "_encoded" for col in categorical_features if col in df_clean.columns
]
feature_columns = available_numerical + available_categorical

X = df_clean[feature_columns]
Y = df_clean["home_covered"]

print(f"Final dataset shape: {X.shape}")
print("Features used:", feature_columns)

# Save the processed data to avoid re-processing next time
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(data_dir, exist_ok=True)

# Create filename with timestamp
data_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
processed_data_filename = f"processed_nfl_data_{data_timestamp}.pkl"
processed_data_path = os.path.join(data_dir, processed_data_filename)

# Save X, Y, and metadata
processed_data = {
    "X": X,
    "Y": Y,
    "feature_columns": feature_columns,
    "label_encoders": le_dict,
    "years_processed": f"{years[0]}-{years[-1]}",
    "games_count": len(X),
    "timestamp": data_timestamp,
}

joblib.dump(processed_data, processed_data_path)
print(f"\nProcessed data saved to: {processed_data_path}")
print(f"Games: {len(X)}, Features: {len(feature_columns)}")

Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(Xtrain, ytrain)

y_pred = model.predict(Xtest)
accuracy = accuracy_score(ytest, y_pred)
print("accuracy:", accuracy)

# Create models directory if it doesn't exist
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)

# Create filename with timestamp and accuracy
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
accuracy_str = f"{accuracy:.4f}".replace(".", "")
model_filename = f"nfl_spread_model_{timestamp}_acc{accuracy_str}.pkl"
model_path = os.path.join(models_dir, model_filename)

# Save the model and encoders
model_data = {
    "model": model,
    "feature_columns": feature_columns,
    "label_encoders": le_dict,
    "accuracy": accuracy,
    "training_samples": len(Xtrain),
    "test_samples": len(Xtest),
    "years_trained": f"{years[0]}-{years[-1]}",
    "timestamp": timestamp,
}

joblib.dump(model_data, model_path)
print(f"\nModel saved to: {model_path}")
print(f"Model accuracy: {accuracy:.4f}")
print(f"Training samples: {len(Xtrain)}")
print(f"Test samples: {len(Xtest)}")
