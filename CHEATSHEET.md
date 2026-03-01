# ⚽ Cheatsheet: EPL Quantitative Prediction System

## 1. The Engineering Logic
| Component | Technical Term | The "Why" |
| :--- | :--- | :--- |
| **Pipeline** | Data Encapsulation | Prevents "Data Leakage" by ensuring the scaler only sees the training data during cross-validation. |
| **TimeSeriesSplit** | Temporal Validation | Traditional K-Fold is biased for sports. We must train on past seasons to predict future seasons. |
| **Random Forest** | Non-Linear Ensemble | Captures complex interactions (e.g., how "Venue" impacts "Rolling Shots") that a linear model misses. |
| **ROC-AUC** | Probabilistic Metric | Measures the model's ability to distinguish between a Win and a non-Win, regardless of the classification threshold. |

## 2. Technical Variables
*   **`rolling_gf`**: The average Goals For over the last 3 matches. Captures "Form."
*   **`venue_code`**: Categorical encoding (Home/Away). 
*   **`n_jobs=-1`**: Parallel processing. Tells your Mac to use all its CPU cores to speed up the Grid Search.

## 3. Interview Pitch
> "I developed a production-ready ML pipeline to predict EPL match outcomes. Unlike standard classifiers, I implemented a **TimeSeriesSplit** strategy to respect chronological integrity. By optimizing a **RandomForest** via **GridSearchCV**, I focused on **ROC-AUC** to ensure the model correctly ranks win probabilities—a critical requirement for building profitable sports trading strategies."