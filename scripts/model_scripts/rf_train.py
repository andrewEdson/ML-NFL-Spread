import nfl_data_py as nfl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime

# -------------------------------
# 1. LOAD DATA
# -------------------------------
years = list(range(1999, 2024))
print("Loading schedules and weekly player stats...")
schedules = nfl.import_schedules(years)
player_stats = nfl.import_weekly_data(years)

# -------------------------------
# 2. TEAM-LEVEL WEEKLY STATS (AGGREGATED)
# -------------------------------
print("Aggregating team stats...")
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
            "sack_fumbles": "sum",
            "passing_epa": "sum",
            "rushing_epa": "sum",
            "receiving_epa": "sum",
            "rushing_fumbles_lost": "sum",
            "receiving_fumbles_lost": "sum",
        }
    )
    .reset_index()
    .rename(columns={"recent_team": "team"})
)

# -------------------------------
# 3. SHIFT STATS BY ONE WEEK TO PREVENT LEAK
# -------------------------------
print("Shifting team stats one week forward to prevent data leaks...")
team_stats = team_stats.sort_values(["team", "season", "week"])

# Shift everything forward by 1 week
cols_to_shift = [
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "fantasy_points",
    "sack_fumbles",
    "passing_epa",
    "rushing_epa",
    "receiving_epa",
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
]
team_stats[cols_to_shift] = team_stats.groupby(["team", "season"])[cols_to_shift].shift(
    1
)

# Fill Week 1 values with zeros (no prior data)
team_stats = team_stats.fillna(0)

# -------------------------------
# 4. ADD ROLLING AVERAGES (LAST 3 WEEKS)
# -------------------------------
print("Computing rolling averages for past 3 weeks...")
rolling_cols = cols_to_shift
team_stats[rolling_cols] = (
    team_stats.groupby(["team", "season"])[rolling_cols]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

# -------------------------------
# 5. MERGE STATS INTO SCHEDULES
# -------------------------------
print("Merging team stats into game schedules...")
schedules = schedules.merge(
    team_stats,
    left_on=["season", "week", "home_team"],
    right_on=["season", "week", "team"],
    suffixes=("", "_home"),
    how="left",
).drop(columns=["team"], errors="ignore")

schedules = schedules.merge(
    team_stats,
    left_on=["season", "week", "away_team"],
    right_on=["season", "week", "team"],
    suffixes=("", "_away"),
    how="left",
).drop(columns=["team"], errors="ignore")

# -------------------------------
# 6. CLEAN + FEATURE ENGINEERING
# -------------------------------
print("Cleaning and creating features...")
df = schedules[
    (schedules["game_type"] == "REG")
    & (schedules["spread_line"].notna())
    & (schedules["away_score"].notna())
].copy()

# Actual spread = home score - away score
df["actual_spread"] = df["home_score"] - df["away_score"]
df["spread_diff"] = df["actual_spread"] - df["spread_line"]
df["home_covered"] = df["spread_diff"] > 0

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
]

team_stat_base = cols_to_shift
team_features = []
for stat in team_stat_base:
    if stat in df.columns:
        team_features.append(stat)
    if f"{stat}_away" in df.columns:
        team_features.append(f"{stat}_away")

numerical_features = base_numerical_features + team_features

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

df_clean = df[numerical_features + categorical_features + ["home_covered"]].copy()

# -------------------------------
# 7. HANDLE MISSING VALUES
# -------------------------------
print("Filling missing values...")
for col in numerical_features:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

for col in categorical_features:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna("Unknown")

# -------------------------------
# 8. ENCODE CATEGORICAL FEATURES
# -------------------------------
print("Encoding categorical variables...")
le_dict = {}
for col in categorical_features:
    le = LabelEncoder()
    df_clean[col + "_encoded"] = le.fit_transform(df_clean[col].astype(str))
    le_dict[col] = le

feature_columns = numerical_features + [c + "_encoded" for c in categorical_features]
X = df_clean[feature_columns]
y = df_clean["home_covered"]

# -------------------------------
# 9. CHRONOLOGICAL SPLIT (NO LEAKS)
# -------------------------------
print("Splitting data chronologically...")
train = df_clean[df_clean["season"] < 2022]
test = df_clean[df_clean["season"] >= 2022]

X_train = train[feature_columns]
y_train = train["home_covered"]
X_test = test[feature_columns]
y_test = test["home_covered"]

# -------------------------------
# 10. TRAIN MODEL
# -------------------------------
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy (2022–2023 Test): {accuracy:.4f}")

# -------------------------------
# 11. SAVE MODEL & METADATA
# -------------------------------
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"rf_nfl_spread_model_{timestamp}_acc{accuracy:.3f}.pkl"
model_path = os.path.join(models_dir, model_filename)

model_data = {
    "model": model,
    "feature_columns": feature_columns,
    "label_encoders": le_dict,
    "accuracy": accuracy,
    "training_years": "1999–2021",
    "testing_years": "2022–2023",
    "timestamp": timestamp,
}
joblib.dump(model_data, model_path)

print(f"\nModel saved to: {model_path}")
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
