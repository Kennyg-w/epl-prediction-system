# ⚽ EPL Prediction System

A machine learning system that predicts English Premier League match outcomes and evaluates betting strategies using model probabilities.

This project includes:

- Feature-engineered match data
- Time-series safe model training
- Hyperparameter tuning
- Probability calibration
- Streamlit prediction UI
- Backtesting engine for betting strategies

---

# 🚀 Project Overview

The goal is to build a **data-driven EPL win prediction model** and evaluate whether the model can outperform bookmaker odds.

The system:

1. Trains a machine learning model on historical match data
2. Generates calibrated win probabilities
3. Determines optimal decision thresholds
4. Simulates betting strategies based on model edge

---

# 🧠 Model

Model used:

**HistGradientBoostingClassifier**

Key characteristics:

- Handles nonlinear relationships
- Efficient on tabular data
- Robust with moderate feature counts

Training includes:

- TimeSeriesSplit cross-validation
- GridSearch hyperparameter tuning
- Youden's J threshold optimization
- Optional probability calibration

---

# 📊 Example Results

| Metric | Value |
|------|------|
| TimeSeries CV ROC-AUC | ~0.64 |
| Test ROC-AUC | ~0.64 |
| Test Accuracy | ~0.63 |
| Best Threshold | ~0.40 |

---

# 📦 Project Structure
