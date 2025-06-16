# Design Document: SLA Predictor

## 1. Introduction

The SLA Predictor is a machine learning-based system designed to predict whether a daily feed, expected to arrive by 5 PM, will meet or miss its Service Level Agreement (SLA) on a given date. The system uses historical SLA outcome data to train a classification model, generating predictions for future dates.

The problem is a **binary classification task**:
- **Input**: A specific date.
- **Output**: "Yes" (SLA met) or "No" (SLA missed).

This design emphasizes simplicity while incorporating additional features to improve prediction accuracy.

## 2. System Architecture

The SLA Predictor includes the following components:
1. **Data Preprocessing**
2. **Feature Extraction**
3. **Model Training**
4. **Prediction**
5. **Evaluation**

### 2.1. Data Preprocessing

The system assumes historical data in a structured format (e.g., CSV) with:
- **Date**: The feed date.
- **SLA Outcome**: Binary label (e.g., 0 = met, 1 = missed).

Preprocessing steps:
- Standardize date format (e.g., YYYY-MM-DD).
- Ensure binary SLA outcomes (0 or 1).
- **Support Lagged Features**: Include consecutive days for features like "Previous Day SLA Missed" and "Consecutive Misses." Set default values (e.g., 0) for the first day.
- **External Data**: Incorporate a holiday calendar and, optionally, workload or staffing data if available.

### 2.2. Feature Extraction

Features extracted from each date include:
- **Day of the Week**: Integer (0 = Monday, 6 = Sunday).
- **Day of the Month**: Integer (1–31).
- **Month**: Integer (1 = January, 12 = December).
- **Week of the Year**: ISO week number (1–53).
- **Day of the Year**: Integer (1–365/366) for seasonal trends.
- **Is Weekend**: Binary (1 = Saturday/Sunday, 0 = weekday).
- **Is Holiday**: Binary (1 = holiday, 0 = non-holiday).
- **Days Since Last Holiday**: Integer counting days since the last holiday (captures backlog effects).
- **Previous Day SLA Missed**: Binary (1 = previous day missed SLA, 0 = met).
- **Consecutive Misses**: Integer counting consecutive SLA misses up to the previous day.
- **Rolling Miss Rate (7-day)**: Float (0–1) capturing recent trend.

These features capture calendar effects, seasonal trends, holiday proximity, and recent SLA streaks without relying on external workload or capacity data. **Note**: Lag-based variables (e.g., "Previous Day SLA Missed," "Consecutive Misses") default to 0 when forecasting future dates.

### 2.3. Model Training

Recommended models:
- **Logistic Regression**: Simple, interpretable, ideal for small datasets.
- **Decision Tree**: Captures non-linear patterns with clear rules.
- **Random Forest**: Boosts accuracy for larger datasets.

Training considerations:
- Begin with calendar features; incorporate lag-based variables (Consecutive Misses, Rolling Miss Rate) when sufficient history is available.
- Start with basic models, adding complexity/features as needed.

### 2.4. Prediction

For a future date:
1. Extract features (e.g., day of the week, holiday status).
2. Set lagged features ("Previous Day SLA Missed," "Consecutive Misses") to defaults or estimates.
3. Use defaults or historical averages for "Workload Volume" and "Staffing Level" if unavailable.
4. Model outputs:
   - Probability > 0.5: "No" (SLA missed).
   - Otherwise: "Yes" (SLA met).

### 2.5. Evaluation

Evaluation process:
- **Data Split**: Time-based (train on earlier data, test on later).
- **Metrics**:
  - **Accuracy**: General performance.
  - **Recall**: Prioritize catching SLA misses if critical.
  - **AUC-ROC**: Overall model quality.

Target: >70% accuracy, adjustable with feature/model tweaks.

## 3. Deployment

- **Interface**: Simple tool (e.g., CLI or web app) for date input and prediction output.
- **Logging**: Track predictions and outcomes for monitoring.

## 4. Retraining and Maintenance

- **Frequency**: Monthly/quarterly updates with new data.
- **Process**: Append new SLA outcomes and retrain.

## 5. Potential Improvements

- **More Features**: Pre-/post-holiday indicators, days to next holiday.
- **Advanced Models**: Gradient boosting (e.g., XGBoost) for better accuracy.
- **External Data**: System outages, weather impacts (if relevant).

## 6. Assumptions

- Historical patterns predict future outcomes.
- Holiday calendar is accurate.
- Dataset is sufficient for training.
- Default values suffice for unavailable data in predictions.

## 7. Constraints

- Limited to date-based and optional external features.
- Dependent on data quality/quantity.
- Lagged/external features may limit future prediction precision.

## 8. Success Criteria

- >70% accuracy on test data.
- Actionable predictions for planning.