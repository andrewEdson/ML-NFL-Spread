# 🏈 NFL Spread Prediction Model Evaluation (1999–2023)

## Overview

The objective of this experiment was to evaluate machine learning models for predicting **NFL game outcomes against the spread**.  
Using the `nfl_data_py` API, historical data from **1999–2022** was used for training, with the **2022–2023 season** reserved as a held-out test set to simulate real-world season performance.

Four models were trained and tested:

- **Random Forest**
- **LightGBM**
- **XGBoost**
- **CatBoost**

The goal was to identify whether a model could provide a consistent statistical **edge against the house (Vegas spread)**.

---

## ⚙️ Model Performance Summary

| Model          | Test Accuracy (2022 Season) |
| -------------- | --------------------------- |
| Random Forest  | **0.540**                   |
| LightGBM       | 0.521                       |
| CatBoost       | 0.521                       |
| XGBoost        | 0.519                       |
| Ensemble (All) | 0.514                       |

---

## 📊 Methodology

- **Training data:** 1999–2022 regular season games
- **Test data:** 2022–2023 season
- **Target variable:** Whether the home team **covered the spread** (`home_covered = True/False`)
- **Features:** Included team-level aggregated player stats, rest days, weather, moneyline, coach, quarterback, and injury-related data (where applicable).
- **Evaluation metric:** Accuracy (% of games correctly predicted against the spread)

This structure mimics how a bettor would make predictions in real time—using only past seasons’ data to forecast the upcoming season.

---

## 🧠 Observations & Insights

### 1. Random Forest Outperformed Boosted Models

The **Random Forest** achieved the highest accuracy at **54%**, slightly above the break-even threshold (50%).  
Interestingly, this model may have benefited from its _randomized feature selection and split criteria_, introducing a degree of noise that helped generalization.

> While boosted models (XGBoost, LightGBM, CatBoost) are designed to optimize feature splits for maximum predictive power, they may have overfit patterns specific to prior seasons rather than generalizable trends.

From a football perspective, this aligns with intuition: Random Forests can “hedge” by averaging over many sub-optimal but diverse decision paths, which may inadvertently capture non-linear, situational factors not easily modeled by gradient boosting.

---

### 2. Boosted Models Showed High Feature Selectivity

Models such as **XGBoost** and **CatBoost** heavily weighted features related to:

- **Injuries**
- **Rest differential**
- **Quarterback performance**
- **Coaching**
- **Offensive production**

These are all intuitive indicators for game outcomes—particularly **offensive consistency** and **leadership quality**.  
However, these features may not be as predictive **against the spread**, where Vegas lines already price in these public metrics.

In essence, the boosted models might be **too "smart"**, keying in on variables Vegas already accounts for.

---

### 3. Ensemble Model Did Not Improve Results

When combining all four models into a simple ensemble, accuracy **fell to 51.4%**.  
This suggests the models were **highly correlated** in their predictions—likely responding to the same statistical signals in the data rather than offering complementary insights.

---

### 4. Away Team Bias

A significant bias emerged across all models: they tended to **predict the away team to cover** more frequently.  
This could indicate:

- A data imbalance (e.g., more away teams historically covering)
- Or, more interestingly, that **Vegas may overvalue home-field advantage** in setting the spread.

This aligns with several academic analyses suggesting that public betting often favors home teams, leading to slightly inflated home spreads.

---

### 5. Vegas is Extremely Efficient

A ~54% accuracy ceiling reinforces that **Vegas spreads are finely tuned** to public perception and statistical expectation.  
Consistently beating the spread is inherently difficult, as it reflects the consensus expectation of thousands of bettors and models—creating an almost efficient market.

---

## 🏟️ Reflections from Experience

As a former **Division I football player**, I can confirm that many “intangibles” — player confidence, coaching adjustments, and momentum — rarely appear in statistical data yet significantly affect outcomes.  
While models can capture trends in **injuries, rest, or offensive stats**, they struggle with **game flow and adaptive strategy**, which coaches and players manage dynamically.

Interestingly, the **boosted models** (XGBoost, CatBoost) seemed to weigh these “real football” features more meaningfully, while the **Random Forest** leaned on less intuitive splits — potentially explaining its slight edge through randomness rather than true predictive understanding.

---

## 🧩 Next Steps

Future research directions could include:

- **Feature engineering:** Adding injury severity scores, EPA/play, and turnover margin
- **Temporal modeling:** Using sequence models (e.g., LSTMs) to capture team momentum
- **Model ensembling with meta-learning:** Weighted stacking or blending of heterogeneous models
- **Expert comparison:** Comparing model picks vs. human (expert) picks vs. random chance

Additionally, testing these models on **moneyline (win/loss)** predictions might yield higher accuracy, as Vegas spreads introduce inherent randomness by design.

---

## 🧾 Conclusion

The results reinforce that while machine learning can **approach parity with Vegas spreads**, the edge remains razor-thin.  
A 54% accuracy rate theoretically provides a small positive return, but sustaining it long-term requires careful feature management, model validation, and constant adaptation to new season dynamics.

Even so, this experiment highlights a critical truth in sports analytics:

> The best models often mirror what experienced players and coaches already know — football outcomes are driven by complex, contextual factors that stats only partially capture.
