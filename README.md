# NFL Spread Prediction with Machine Learning

A comprehensive machine learning project for predicting NFL point spread outcomes using advanced feature engineering, ensemble methods, and explainable AI techniques. The system processes 25+ years of historical data and achieves 53.96% accuracy through proper chronological validation and data leakage prevention.

## Overview

This project leverages multiple gradient boosting and ensemble algorithms to predict whether the home team will cover the betting spread in NFL games. The models utilize:

- Team offensive and defensive performance statistics
- Historical rolling averages and trend analysis
- Position-specific injury impact assessment (2009-present)
- Betting market signals and context
- Personnel factors (coaches, quarterbacks, referees)
- Environmental conditions and rest patterns
- SHAP-based model explainability

## Technical Highlights

- **Multi-model ensemble architecture** combining 4 specialized algorithms
- **78+ engineered features** with time-series aware preprocessing
- **Chronological train/test validation** (1999-2021 train, 2022-2023+ test)
- **Data leakage prevention** through time-shifted feature engineering
- **SHAP explainability framework** for model transparency
- **Production-ready pipeline** for real-time season predictions

## Installation

```bash
# Clone repository
git clone <repository-url>
cd ML-NFL-Spread

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Models

```bash
# Train individual models
python scripts/model_scripts/xgb_train.py       # XGBoost
python scripts/model_scripts/rf_train.py        # Random Forest
python scripts/model_scripts/lgbm_train.py      # LightGBM
python scripts/model_scripts/cat_boost_train.py # CatBoost
```

### Model Evaluation

```bash
# Compare all models with feature importance analysis
python evaluation/eval.py

# Generate SHAP explainability visualizations
python evaluation/shap_eval.py

# Run ensemble predictions
python evaluation/ensemble_test.py
```

### Generate Test Data for New Season

```bash
# Create 2025 season test dataset with all engineered features
python scripts/data_scripts/2025_test_data.py
```

## Model Performance

| Model              | Accuracy | Training Period | Test Period | Specialized Focus             |
| ------------------ | -------- | --------------- | ----------- | ----------------------------- |
| Random Forest      | 53.96%   | 1999-2021       | 2022-2023   | Personnel & coaching factors  |
| LightGBM           | 52.12%   | 1999-2021       | 2022-2023   | Gradient boosting efficiency  |
| CatBoost           | 52.12%   | 1999-2021       | 2022-2023   | Native categorical handling   |
| XGBoost            | 51.93%   | 1999-2021       | 2022-2023   | Feature interaction discovery |
| Ensemble (4-model) | 51.38%   | All Models      | 2022-2023   | Averaged predictions          |

### Advanced Analytics Integration

- **SHAP Analysis**: Model explainability with feature contribution visualization
- **Ensemble Methods**: 4-model averaging for robust predictions
- **Injury Integration**: Position-specific injury severity mapping (2009-present)
- **Feature Importance Reporting**: Automatic top-20 feature ranking after training
- **Feature Diversity**: Zero overlap in top features across models indicates complementary learning

Note: Chronological validation ensures no data leakage - models predict genuine future games.

## Data Leakage Discovery and Resolution

### Problem Identification

Initial model development achieved approximately 78% accuracy using random train/test splits. Investigation revealed critical data leakage:

- Team statistics from the same week as the predicted game were used as features
- Random splitting allowed models to "peek" at future information
- Performance metrics were unrealistically high and not production-viable

### Solution Implementation

1. **Time-shifted features**: All team statistics shifted by 1 week (only historical performance used)
2. **Chronological validation**: Training on 1999-2021, testing on 2022-2023 (strict temporal separation)
3. **Rolling averages**: 3-week rolling means replace single-game statistics

Result: Realistic 53% accuracy representing genuine predictive capability for future games.

## Architecture

### Data Processing Pipeline

1. **Raw Data Ingestion**: NFL game schedules and player statistics via nfl-data-py library
2. **Injury Data Integration**: Position-specific injury mapping with severity weights (2009-present)
3. **Team-Level Aggregation**: Player statistics aggregated to team-level weekly performance
4. **Temporal Feature Engineering**: Time-shifted features prevent data leakage
5. **Rolling Statistics**: 3-week rolling averages capture performance trends
6. **Feature Engineering**: 78+ features including injury metrics, betting data, and personnel factors

### Multi-Model Architecture

**Four Specialized Algorithms:**

1. **Random Forest**: Ensemble learning focused on personnel factors (coaches, QBs, referees)
2. **LightGBM**: Gradient boosting with efficient feature interaction discovery
3. **CatBoost**: Native categorical variable handling without encoding requirements
4. **XGBoost**: Advanced gradient boosting with hyperparameter optimization

**Ensemble Methodology:**

- Arithmetic averaging of all 4 model predictions
- Categorical feature decoding ensures CatBoost compatibility
- Validated on 543 real test games from 2022-2023 seasons

### Model-Specific Feature Prioritization

**Random Forest:**

- Personnel factors (QB names, coaches, referees - encoded)
- Betting market signals (total line, spread line)
- Advanced efficiency metrics (EPA - Expected Points Added)
- Rushing performance indicators

**XGBoost:**

- Receiving and passing game metrics
- Turnover statistics (interceptions, fumbles)
- Position-specific injury impact factors
- Coaching influence indicators

**LightGBM:**

- Personnel factors (encoded QB and coach features)
- Team efficiency metrics (passing EPA, rushing yards)
- Betting market context (total line information)
- Game flow indicators (fantasy points, EPA metrics)

**CatBoost:**

- Raw categorical variables (QB names, coaches, referees)
- Team identity indicators (home/away team)
- Efficiency metrics (EPA, rushing performance)
- Betting market context (spread and total lines)

### Technical Architecture

- Multi-model ensemble with 4 specialized algorithms
- 78 engineered features including injury metrics and betting data
- Chronological train/test split (1999-2021 train, 2022-2023+ test)
- Time-shifted features prevent data leakage
- SHAP framework for model explainability
- Dual categorical handling (encoding and native approaches)

## 📁 Project Structure

```
## Project Structure

```

ML-NFL-Spread/
├── scripts/
│ ├── model_scripts/
│ │ ├── xgb_train.py # XGBoost training with feature importance
│ │ ├── rf_train.py # Random Forest training
│ │ ├── lgbm_train.py # LightGBM with top-20 feature reporting
│ │ └── cat_boost_train.py # CatBoost training
│ ├── data_scripts/
│ │ ├── download_data.py # Historical data preprocessing pipeline
│ │ └── 2025_test_data.py # 2025 season test dataset generation
│ └── helpful/
│ ├── train.py # Legacy training script
│ ├── train_from_processed.py # Fast training from saved data
│ └── columns.py # Data exploration utilities
├── evaluation/
│ ├── eval.py # Multi-model comparison
│ ├── shap_eval.py # SHAP explainability analysis
│ ├── ensemble_test.py # 4-model ensemble evaluation
│ └── 2025_test.py # 2025 season evaluation
├── models/ # Trained model files (.pkl, .cbm)
├── data/ # Processed datasets (gitignored)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

```

## Key Components

### Training Scripts

- **xgb_train.py**: XGBoost implementation with hyperparameter optimization
- **rf_train.py**: Random Forest with personnel-focused feature engineering
- **lgbm_train.py**: LightGBM with automatic top-20 feature importance reporting
- **cat_boost_train.py**: CatBoost with native categorical variable handling

### Data Processing Scripts

- **download_data.py**: Complete historical data preprocessing pipeline (1999-2024)
- **2025_test_data.py**: Generates test dataset for 2025 season predictions with all engineered features

### Analysis Scripts

- **eval.py**: Comprehensive multi-model feature importance comparison
- **shap_eval.py**: SHAP explainability analysis with visualizations
- **ensemble_test.py**: 4-model ensemble evaluation on real test data
- **2025_test.py**: Evaluation framework for current season predictions
- **columns.py**: Data exploration and column analysis utilities

## Feature Engineering

### Data Leakage Prevention

- **Time-shifted features**: Only historical performance used (1-week shift)
- **Rolling averages**: 3-week rolling means capture performance trends
- **Chronological splits**: Strict temporal separation (train on past, test on future)

### Engineered Features

- **Injury Analytics**: Position-specific injury severity mapping (QB=5, RB=3, etc.)
- **Team Statistics**: Player-level data aggregated to team-level with EPA metrics
- **Betting Context**: Spreads, moneylines, totals reflecting market efficiency
- **Environmental Factors**: Weather, temperature, wind conditions
- **Personnel Indicators**: Coaches, referees, quarterbacks (encoded and categorical)
- **Rest and Travel**: Days between games, travel distance patterns

### Categorical Feature Handling

- **Encoding Approach**: LabelEncoder for XGBoost, Random Forest, and LightGBM
- **Native Approach**: CatBoost processes categorical features without encoding
- **Ensemble Compatibility**: Automatic categorical decoding for CatBoost integration

## Model Insights and Feature Analysis

### Multi-Model Learning Strategy

Analysis reveals zero overlap in top-10 features across all four models, demonstrating complementary learning patterns optimal for ensemble approaches:

**Random Forest Strategy:**
- Personnel-driven analysis (QB performance, coaching impact, referee patterns)
- Betting market context (lines and spreads)
- Advanced efficiency metrics (EPA across all phases)

**XGBoost Strategy:**
- Game flow dynamics (receiving performance, turnover statistics)
- Position-specific injury impact assessment
- Situational advantages (coaching, rest patterns)

**LightGBM Strategy:**
- EPA-based efficiency metrics across game phases
- Encoded personnel factors (QB and coaching features)
- Betting market signals (total line information)

**CatBoost Strategy:**
- Native categorical processing (unencoded QB names, coaches, referees)
- Team identity indicators
- Efficiency metrics (EPA, rushing performance)

### Explainability Framework

- **SHAP Analysis**: Contribution values for every prediction
- **Feature Transparency**: Interpretable model decision-making
- **Comparative Visualization**: Feature importance differences across models
- **Permutation Importance**: Alternative analysis for model validation

### Key Finding

Zero feature overlap in top-10 features across all models indicates complementary learning - each algorithm extracts value from different aspects of NFL games, validating the ensemble methodology.

## Future Development

- Performance-based weighted ensemble (replace simple averaging)
- Advanced injury impact prediction models
- Real-time prediction pipeline for live season

## Results Summary

This project demonstrates advanced NFL spread prediction capabilities:

- 53.96% accuracy with Random Forest (best individual model)
- Four-model ensemble with complementary learning strategies
- SHAP explainability framework for model transparency
- Position-specific injury severity integration (2009-present)
- Rigorous chronological validation preventing overfitting
- Production-ready pipeline with automated feature importance reporting
- 78+ engineered features from 25+ years of NFL data
- Current season test data generation capability

Note: 53.96% accuracy represents a significant edge over the break-even threshold of approximately 52.4% required for profitable sports betting.

## Dependencies

See requirements.txt for complete list. Core packages:

- **Data Processing**: nfl-data-py, pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, lightgbm, catboost
- **Analysis and Visualization**: matplotlib, seaborn, shap
- **Utilities**: joblib, fastparquet, pickle

## Technical Skills Demonstrated

- Machine learning model development and ensemble methods
- Time-series feature engineering with leakage prevention
- Data pipeline development and automation
- Model explainability and interpretability (SHAP)
- Performance optimization and hyperparameter tuning
- Production-ready code architecture and documentation

## License

This project is available for educational and portfolio purposes.

---

**Disclaimer**: This project is intended for educational and research purposes only. Sports betting involves financial risk and should be approached responsibly.
```
