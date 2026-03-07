# ⚽ Cheatsheet: EPL Quantitative Prediction System

Quick reference for the architecture, ML decisions, and interview explanation of the project.

---

# 1. The Engineering Logic

| Component | Technical Term | Why It Matters |
|-----------|---------------|---------------|
| **Feature Pipeline** | Data Engineering | Rolling match statistics capture team form and recent performance trends. |
| **TimeSeriesSplit** | Temporal Validation | Prevents future data leakage. The model only learns from past matches to predict future ones. |
| **HistGradientBoosting** | Gradient Boosted Trees | Strong performance on tabular datasets and handles nonlinear relationships effectively. |
| **GridSearchCV** | Hyperparameter Optimization | Searches for the best model configuration using cross-validation. |
| **ROC-AUC** | Probabilistic Evaluation Metric | Measures how well the model ranks winning vs non-winning matches independent of threshold. |
| **Youden's J Statistic** | Optimal Threshold Selection | Chooses the classification threshold that balances sensitivity and specificity. |
| **Permutation Importance** | Model Explainability | Measures how much each feature contributes to prediction accuracy. |

---

# 2. Key Feature Engineering

| Feature | Meaning |
|--------|--------|
| `gf_rolling` | Rolling average of goals scored over previous matches |
| `ga_rolling` | Rolling average of goals conceded |
| `gd_rolling` | Goal difference trend (gf − ga) |
| `form_rolling` | Recent team performance |
| `team_avg_points` | Long-term team strength |
| `team_avg_gd` | Long-term goal difference strength |
| `rest_days` | Recovery time between matches |

Opponent features mirror the same structure:
