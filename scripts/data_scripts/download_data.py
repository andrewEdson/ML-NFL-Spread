import nfl_data_py as nfl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime
from xgboost import XGBClassifier

# -------------------------------
# 1. LOAD DATA
# -------------------------------
years = list(range(1999, 2024))
print("Loading schedules and weekly player stats...")
schedules = nfl.import_schedules(years)
player_stats = nfl.import_weekly_data(years)

# Injury data is only available from 2009 onwards
injury_years = list(range(2009, 2024))
print("Loading injury data (2009-2023)...")
injuries = nfl.import_injuries(injury_years)

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
# 3.5. INJURY DATA PROCESSING
# -------------------------------
print("Processing injury data...")

# Map injury status to severity scores
injury_status_mapping = {
    "Out": 4,
    "Doubtful": 3,
    "Questionable": 2,
    "Probable": 1,
    "IR": 5,  # Injured Reserve
    "PUP": 5,  # Physically Unable to Perform
    "NFI": 4,  # Non-Football Injury
    "SUSP": 3,  # Suspended
    "DNR": 2,  # Did Not Report
    "COV": 1,  # COVID-19 list
}

# Clean and process injury data
injuries_clean = injuries.copy()
injuries_clean["injury_severity"] = (
    injuries_clean["report_status"].map(injury_status_mapping).fillna(0)
)

# Group key positions for analysis
position_groups = {
    "QB": ["QB"],
    "RB": ["RB", "FB"],
    "WR": ["WR"],
    "TE": ["TE"],
    "OL": ["C", "G", "T"],
    "DL": ["DE", "DT", "NT"],
    "LB": ["LB", "ILB", "OLB", "MLB"],
    "DB": ["CB", "S", "SS", "FS", "DB"],
    "ST": ["K", "P", "LS"],
}

# Create position group mapping
pos_group_map = {}
for group, positions in position_groups.items():
    for pos in positions:
        pos_group_map[pos] = group

injuries_clean["position_group"] = (
    injuries_clean["position"].map(pos_group_map).fillna("OTHER")
)

# Aggregate injury data by team, season, week
injury_stats = (
    injuries_clean.groupby(["team", "season", "week"])
    .agg(
        {
            "injury_severity": ["count", "sum", "mean"],
            "gsis_id": "nunique",  # Number of unique injured players
        }
    )
    .round(2)
)

# Flatten column names
injury_stats.columns = [
    "total_injuries",
    "injury_severity_sum",
    "avg_injury_severity",
    "injured_players",
]
injury_stats = injury_stats.reset_index()

# Add position-specific injury counts
position_injury_stats = (
    injuries_clean.groupby(["team", "season", "week", "position_group"])
    .agg({"injury_severity": "sum"})
    .unstack(fill_value=0)
    .round(2)
)

position_injury_stats.columns = [
    f"injuries_{pos.lower()}"
    for pos in position_injury_stats.columns.get_level_values(1)
]
position_injury_stats = position_injury_stats.reset_index()

# Merge injury stats
injury_stats = injury_stats.merge(
    position_injury_stats, on=["team", "season", "week"], how="left"
)

# Fill missing values with 0 (no injuries reported)
injury_cols = [
    col for col in injury_stats.columns if col not in ["team", "season", "week"]
]
injury_stats[injury_cols] = injury_stats[injury_cols].fillna(0)

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

# Add rolling averages for injury data
print("Computing rolling averages for injury data...")
injury_rolling_cols = [
    col for col in injury_stats.columns if col not in ["team", "season", "week"]
]

# Shift injury data by 1 week to prevent data leakage (same as team stats)
injury_stats[injury_rolling_cols] = injury_stats.groupby(["team", "season"])[
    injury_rolling_cols
].shift(1)
injury_stats = injury_stats.fillna(0)

# Apply rolling averages to injury data
injury_stats[injury_rolling_cols] = (
    injury_stats.groupby(["team", "season"])[injury_rolling_cols]
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

# Merge injury data for home team
print("Merging injury data into game schedules...")
schedules = schedules.merge(
    injury_stats,
    left_on=["season", "week", "home_team"],
    right_on=["season", "week", "team"],
    suffixes=("", "_home"),
    how="left",
).drop(columns=["team"], errors="ignore")

# Merge injury data for away team
schedules = schedules.merge(
    injury_stats,
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
]

team_stat_base = cols_to_shift
team_features = []
for stat in team_stat_base:
    if stat in df.columns:
        team_features.append(stat)
    if f"{stat}_away" in df.columns:
        team_features.append(f"{stat}_away")

# Add injury features
injury_features = []
injury_base_features = [
    "total_injuries",
    "injury_severity_sum",
    "avg_injury_severity",
    "injured_players",
    "injuries_qb",
    "injuries_rb",
    "injuries_wr",
    "injuries_te",
    "injuries_ol",
    "injuries_dl",
    "injuries_lb",
    "injuries_db",
    "injuries_st",
    "injuries_other",
]

for injury_stat in injury_base_features:
    if injury_stat in df.columns:
        injury_features.append(injury_stat)
    if f"{injury_stat}_away" in df.columns:
        injury_features.append(f"{injury_stat}_away")

numerical_features = base_numerical_features + team_features + injury_features

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
print(f"Available seasons: {sorted(df_clean['season'].unique())}")
print(f"Total games before split: {len(df_clean)}")

train = df_clean[df_clean["season"] < 2022]
test = df_clean[df_clean["season"] >= 2022]

print(f"Training games: {len(train)} (seasons < 2022)")
print(f"Test games: {len(test)} (seasons >= 2022)")

if len(train) == 0:
    print("ERROR: No training data found!")
    exit()
if len(test) == 0:
    print("ERROR: No test data found!")
    exit()

X_train = train[feature_columns]
y_train = train["home_covered"]
X_test = test[feature_columns]
y_test = test["home_covered"]

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Check for any remaining NaN values
if X_train.isna().any().any():
    print("WARNING: NaN values found in X_train")
    print(X_train.isna().sum())

if X_test.isna().any().any():
    print("WARNING: NaN values found in X_test")
    print(X_test.isna().sum())

# -------------------------------
# 10. SAVE PROCESSED DATA
print("Saving processed data and encoders...")
data_file = "processed_nfl_data.pkl"
data_path = os.path.join("data", data_file)

data = {
    "X_train": X_train,
    "y_train": y_train,
    "X_test": X_test,
    "y_test": y_test,
    "feature_columns": feature_columns,
    "label_encoders": le_dict,
}
joblib.dump(data, data_path)
print(f"Processed data saved to {data_path}")
