import pandas as pd
import numpy as np


def simulate_betting(df, probs, threshold, odds_column="bookmaker_odds"):
    """
    Simulates flat betting strategy:
    Bet when predicted probability >= threshold.
    """

    df = df.copy()
    df["pred_prob"] = probs
    df["bet"] = df["pred_prob"] >= threshold

    # Implied probability
    df["implied_prob"] = 1 / df[odds_column]

    # Profit calculation (1 unit stake)
    df["profit"] = np.where(
        df["bet"] & (df["target"] == 1),
        df[odds_column] - 1,
        np.where(df["bet"], -1, 0)
    )

    total_profit = df["profit"].sum()
    roi = total_profit / df["bet"].sum() if df["bet"].sum() > 0 else 0

    return {
        "total_bets": int(df["bet"].sum()),
        "total_profit": round(float(total_profit), 2),
        "roi": round(float(roi), 4)
    }