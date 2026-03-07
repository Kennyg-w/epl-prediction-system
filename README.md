# ⚽ EPL Prediction System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-green)
![UI](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

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

## 🏗 System Architecture

```text
                ┌────────────────────────┐
                │  Raw Match Data        │
                │  processed_matches.csv │
                └─────────────┬──────────┘
                              │
                              ▼
                ┌────────────────────────┐
                │  Feature Engineering   │
                │  prepare_features()    │
                └─────────────┬──────────┘
                              │
                              ▼
                ┌────────────────────────┐
                │  Model Training        │
                │  HistGradientBoosting  │
                │  GridSearchCV          │
                └─────────────┬──────────┘
                              │
                              ▼
                ┌────────────────────────┐
                │  Model Artifacts       │
                │  model + threshold     │
                └───────┬─────────┬──────┘
                        │         │
                        ▼         ▼
        ┌───────────────────┐   ┌──────────────────┐
        │ Streamlit App     │   │ Backtest Engine  │
        │ Prediction UI     │   │ ROI Simulation   │
        └───────────────────┘   └──────────────────┘
```
---

## ⭐ Key Features

- **Time-series aware training**
  - Uses `TimeSeriesSplit` to prevent future data leakage.

- **Feature engineering pipeline**
  - Rolling team statistics
  - Opponent strength metrics
  - Rest advantage features
  - Home advantage interaction (`home_gd_diff`)

- **Advanced model training**
  - `HistGradientBoostingClassifier`
  - Hyperparameter tuning with `GridSearchCV`

- **Probability optimization**
  - Youden's J threshold selection
  - Optional probability calibration

- **Interactive prediction interface**
  - Streamlit app for real-time predictions.

- **Backtesting engine**
  - Simulates betting strategies using model edge
  - Calculates ROI and drawdown.

- **Modular ML system**
  - Training
  - Prediction UI
  - Model artifacts
  - Backtesting CLI

## 🧠 Model

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

## 📊 Example Results

| Metric | Value |
|------|------|
| TimeSeries CV ROC-AUC | ~0.64 |
| Test ROC-AUC | ~0.64 |
| Test Accuracy | ~0.63 |
| Best Threshold | ~0.40 |

---

## 📦 Project Structure

epl-prediction-system
│
├── app
│ └── streamlit_app.py # Streamlit prediction UI
│
├── data
│ ├── raw
│ │ └── processed_matches.csv
│ └── odds # bookmaker odds (future step)
│
├── models
│ └── model artifacts
│
├── notebooks
│ └── exploration.ipynb
│
├── src
│ ├── train.py # model training pipeline
│ ├── artifacts.py # load model bundle
│ ├── backtest.py # betting simulation engine
│ ├── backtest_run.py # CLI backtest runner
│ └── odds.py # odds utilities
│
├── CHEATSHEET.md # developer quick guide
├── README.md
└── requirements.txt

---