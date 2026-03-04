# src/train.py
# Run from project root:
#   python -m src.train
#
# What this script does:
# - Loads data (chronological)
# - Builds leakage-safe features (team + opponent rolling form/strength)
# - Time-aware split (first 80% train, last 20% test)
# - Trains HistGradientBoostingClassifier with TimeSeriesSplit + GridSearchCV (ROC-AUC)
# - Prints permutation importance (model-agnostic)
# - Calibrates probabilities (isotonic)
# - Tunes decision threshold using Youden's J (TPR - FPR)
# - Prints metrics + probability buckets
# - Saves a bundle: {"model": ..., "threshold": ..., "features": [...]}

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from .config import DATA_DIR, MODEL_PATH


def load_data() -> pd.DataFrame:
    """Load processed matches and sort chronologically."""
    file_path = DATA_DIR / "raw" / "processed_matches.csv"
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Leakage-safe feature builder.

    Adds:
    - Rest days (team + opponent)
    - Expanding strength (team + opponent): avg points & avg goal-diff to date (past matches only)
    - Rolling form/goal-diff (past matches only)
    - Opponent rolling/strength via same-date opponent merge
    - Difference features (team - opponent): often boosts ROC-AUC

    Expected columns include:
      date, team, opponent, result, target, venue_code, day_code,
      gf, ga, gf_rolling, ga_rolling, opp_code
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # --- Base outcome signals (safe if shifted before expanding/rolling) ---
    df["points"] = df["result"].map({"W": 3, "D": 1, "L": 0}).astype(float)
    df["match_gd"] = (df["gf"] - df["ga"]).astype(float)

    # --- Rest days (team) ---
    df["rest_days"] = (
        df.groupby("team")["date"]
        .diff()
        .dt.days
        .fillna(7)
        .clip(lower=0, upper=30)
        .astype(float)
    )

    # --- Expanding strength (TEAM) using past matches only ---
    df["team_avg_points"] = (
        df.groupby("team")["points"]
        .shift(1)
        .expanding(min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(float)
    )
    df["team_avg_gd"] = (
        df.groupby("team")["match_gd"]
        .shift(1)
        .expanding(min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(float)
    )

    # --- Rolling form (TEAM) using past matches only ---
    df["form_rolling"] = (
        df.groupby("team")["points"]
        .shift(1)
        .rolling(5, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(float)
    )

    # --- Rolling goal-diff derived from rolling gf/ga ---
    df["gd_rolling"] = (df["gf_rolling"] - df["ga_rolling"]).astype(float)

    # --- Opponent features (same-date merge) ---
    opp_stats = df[[
        "date",
        "team",
        "gf_rolling",
        "ga_rolling",
        "gd_rolling",
        "form_rolling",
        "rest_days",
        "team_avg_points",
        "team_avg_gd",
    ]].copy()

    opp_stats = opp_stats.rename(columns={
        "team": "opponent",
        "gf_rolling": "opp_gf_rolling",
        "ga_rolling": "opp_ga_rolling",
        "gd_rolling": "opp_gd_rolling",
        "form_rolling": "opp_form_rolling",
        "rest_days": "opp_rest_days",
        "team_avg_points": "opp_team_avg_points",
        "team_avg_gd": "opp_team_avg_gd",
    })

    df = df.merge(opp_stats, on=["date", "opponent"], how="left")

    # --- Difference features (TEAM - OPP) ---
    df["form_diff"] = df["form_rolling"] - df["opp_form_rolling"]
    df["gd_diff"] = df["gd_rolling"] - df["opp_gd_rolling"]
    df["avg_points_diff"] = df["team_avg_points"] - df["opp_team_avg_points"]
    df["avg_gd_diff"] = df["team_avg_gd"] - df["opp_team_avg_gd"]
    df["rest_diff"] = df["rest_days"] - df["opp_rest_days"]
    df["home_gd_diff"] = df["venue_code"] * df["gd_diff"]

    # --- Final feature set (pre-match safe) ---
    features = [
        "venue_code",
        "day_code",
        "opp_code",

        # Team rolling/performance
        "gf_rolling",
        "ga_rolling",
        "gd_rolling",
        "form_rolling",

        # Team expanding strength
        "team_avg_points",
        "team_avg_gd",

        # Team rest
        "rest_days",

        # Opponent rolling/performance
        "opp_gf_rolling",
        "opp_ga_rolling",
        "opp_gd_rolling",
        "opp_form_rolling",

        # Opponent expanding strength
        "opp_team_avg_points",
        "opp_team_avg_gd",

        # Opponent rest
        "opp_rest_days",

        # Matchup differences (often strongest)
        "form_diff",
        "gd_diff",
        "home_gd_diff",
        "avg_points_diff",
        "avg_gd_diff",
        "rest_diff",
    ]

    # Drop rows missing engineered features (early season / insufficient history)
    df = df.dropna(subset=features + ["target"]).reset_index(drop=True)

    X = df[features].astype(float)
    y = df["target"].astype(int)

    return X, y


def time_aware_split(X: pd.DataFrame, y: pd.Series, train_frac: float = 0.8):
    """Split first train_frac rows as train; remainder as test."""
    split_idx = int(len(X) * train_frac)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def train_model_gridsearch(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train HistGradientBoostingClassifier with TimeSeriesSplit GridSearchCV
    optimizing ROC-AUC.
    """
    tscv = TimeSeriesSplit(n_splits=5)

    base = HistGradientBoostingClassifier(
        random_state=42,
        early_stopping=True,
    )

    param_grid = {
       "learning_rate": [0.06, 0.08, 0.10],
       "max_depth": [2, 3],
       "max_iter": [400, 800],
       "min_samples_leaf": [5, 10, 20],
       "l2_regularization": [0.0, 0.1, 0.2],
  }

    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    print("Best Params:", grid.best_params_)
    print(f"Best TimeSeries CV ROC-AUC: {grid.best_score_:.4f}")

    return best


def permutation_feature_importance(model, X: pd.DataFrame, y: pd.Series):
    """
    Model-agnostic feature importance using permutation importance.
    Uses ROC-AUC as the scoring metric.
    """
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=42,
        scoring="roc_auc",
        n_jobs=-1,
    )

    importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)

    print("\nTop Permutation Importances (ROC-AUC drop):")
    print(importances.head(15))

    return importances


def calibrate_model(base_model, X_train: pd.DataFrame, y_train: pd.Series):
    """Calibrate probabilities for improved reliability."""
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)
    print("Calibrated model fitted (isotonic).")
    return calibrated


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluate:
    - ROC-AUC (threshold-free)
    - threshold tuned using Youden's J (TPR - FPR)
    - confusion matrix / classification report
    - probability buckets (calibration check)
    """
    probs = model.predict_proba(X_test)[:, 1]

    # Threshold tuning via Youden's J
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores)) if len(thresholds) else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5

    preds = (probs >= best_threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"Best threshold (Youden's J): {best_threshold:.3f}")
    print(f"Test Accuracy (thresholded): {acc:.4f}")
    print(f"Test ROC-AUC (probability): {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Probability buckets
    bins = pd.cut(
        probs,
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        include_lowest=True,
    )
    bucket_df = pd.DataFrame({"prob": probs, "y": y_test.values, "bucket": bins})
    bucket_summary = bucket_df.groupby("bucket", observed=False).agg(
        count=("y", "size"),
        win_rate=("y", "mean"),
        avg_prob=("prob", "mean"),
    )

    print("\nProbability Buckets (Calibration Check):")
    print(bucket_summary)

    return probs, preds, best_threshold


def save_model(model_raw, model_calibrated, threshold_raw: float, threshold_calibrated: float, features: list[str]):
    os.makedirs(MODEL_PATH.parent, exist_ok=True)

    joblib.dump(
        {
            "model_raw": model_raw,
            "model_calibrated": model_calibrated,
            "threshold_raw": threshold_raw,
            "threshold_calibrated": threshold_calibrated,
            "features": features,
        },
        MODEL_PATH,
    )

    print("Saved raw + calibrated models, thresholds, and feature list successfully!")


def main():
    print("Loading data...")
    df = load_data()

    print("Preparing features...")
    X, y = prepare_features(df)

    print("Splitting data (time-aware)...")
    X_train, X_test, y_train, y_test = time_aware_split(X, y, train_frac=0.8)

    print("Training model (GridSearchCV, ROC-AUC) [HGB]...")
    best_model = train_model_gridsearch(X_train, y_train)

    # Permutation importance on training data (raw model)
    permutation_feature_importance(best_model, X_train, y_train)

    print("\nEvaluating RAW model...")
    _, _, threshold_raw = evaluate_model(best_model, X_test, y_test)

    print("\nFitting calibrated model (isotonic)...")
    calibrated_model = calibrate_model(best_model, X_train, y_train)

    print("\nEvaluating CALIBRATED model...")
    _, _, threshold_cal = evaluate_model(calibrated_model, X_test, y_test)

    print("Saving model bundle...")
    save_model(best_model, calibrated_model, threshold_raw, threshold_cal, list(X_train.columns))


if __name__ == "__main__":
    main()