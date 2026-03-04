# src/backtest_run.py
"""
Run a simple betting backtest.

Usage (from project root):
  PYTHONPATH=. python -m src.backtest_run --mode raw --edge 0.05
"""

import argparse
import pandas as pd

from src.artifacts import load_artifact
from src.backtest import pick_edge_threshold_by_roi, simulate_flat_betting
from src.config import DATA_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["raw", "calibrated"], default="raw")
    parser.add_argument("--edge", type=float, default=0.05)
    parser.add_argument("--stake", type=float, default=1.0)
    parser.add_argument("--odds_col", type=str, default="odds")
    args = parser.parse_args()

    # Load raw dataset
    df_raw = pd.read_csv(DATA_DIR / "raw" / "processed_matches.csv")
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw.sort_values("date").reset_index(drop=True)

    # Load model artifact (gives feature list)
    model, threshold, features = load_artifact(args.mode)

    # IMPORTANT: build engineered features the same way training did
    from src.train import prepare_features

    X_all, y_all = prepare_features(df_raw)

    # Align df to X rows (prepare_features drops early NaNs)
    df = df_raw.loc[X_all.index].copy()
    df["target"] = y_all.values  # ensure target aligns

    # Predict probabilities
    df["model_prob"] = model.predict_proba(X_all)[:, 1]


    # Check if odds exist
    if args.odds_col not in df.columns:
        print(f"\n❌ No odds column found: '{args.odds_col}'")
        print("Your dataset currently has NO bookmaker odds.")
        print("Add an odds column (e.g., 'odds') then rerun this backtest.\n")
        print("Columns available:\n", list(df.columns))
        return

    # ROI sweep table
    roi_table = pick_edge_threshold_by_roi(
        df=df,
        prob_col="model_prob",
        odds_col=args.odds_col,
        stake=args.stake,
    )
    print("\nROI table (best first):")
    print(roi_table.head(10))

    # Simulate strategy at chosen edge
    res = simulate_flat_betting(
        df=df,
        prob_col="model_prob",
        odds_col=args.odds_col,
        edge_threshold=args.edge,
        stake=args.stake,
    )

    print("\nStrategy summary:")
    print(res["summary"])


if __name__ == "__main__":
    main()