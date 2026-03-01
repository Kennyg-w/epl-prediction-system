# ⚽ 11. EPL Match Win Prediction: Quantitative ML System

## 📊 Project Overview
A sophisticated machine learning system designed to forecast English Premier League (EPL) match outcomes. This project moves beyond basic classification by implementing a time-series aware production pipeline.

## 🛠️ Technical Achievements
- **Chronological Data Integrity:** Implemented a `date`-based sorting and splitting strategy to eliminate "Future Leakage" in sports forecasting.
- **Production Pipeline:** Built a Scikit-Learn `Pipeline` integrating `ColumnTransformer` and `StandardScaler` for reproducible feature engineering.
- **Advanced Hyperparameter Tuning:** Utilized `GridSearchCV` with `TimeSeriesSplit` (5-fold) to optimize a RandomForest model for high-variance environments.
- **Interpretability:** Extracted feature importance to quantify the impact of "Rolling Form" (last 3 games) vs. "Historical Venue Performance."
- **Model Persistence:** Integrated `joblib` for model serialization, enabling deployment-ready inference.

## 📈 Evaluation Metrics
- **ROC-AUC:** [Insert your score, e.g., 0.63]
- **Precision (Win):** [Insert your score, e.g., 0.58]

## 🐍 Tech Stack
- Python, Scikit-Learn, Pandas, Joblib.