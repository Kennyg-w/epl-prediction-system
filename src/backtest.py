# src/backtest.py
from __future__ import annotations

import numpy as np
import pandas as pd


def implied_prob_from_decimal_odds(odds: pd.Series) -> pd.Series:
    """Convert decimal odds to implied probability."""
    odds = pd.to_numeric(odds, errors="coerce")
    return 1.0 / odds


def simulate_flat_betting(
    df: pd.DataFrame,
    prob_col: str,
    target_col: str = "target",
    odds_col: str = "odds",
    edge_threshold: float = 0.05,
    stake: float = 1.0,
) -> dict:
    """
    Flat betting strategy:
    - Bet stake units when (model_prob - implied_prob) >= edge_threshold
    - Profit:
        win  -> stake*(odds-1)
        lose -> -stake
        no bet -> 0
    Returns summary stats + per-row results dataframe.
    """
    out = df.copy()

    # Clean inputs
    out[prob_col] = pd.to_numeric(out[prob_col], errors="coerce")
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out[odds_col] = pd.to_numeric(out[odds_col], errors="coerce")

    out["implied_prob"] = implied_prob_from_decimal_odds(out[odds_col])
    out["edge"] = out[prob_col] - out["implied_prob"]

    # Place bets where edge is big enough and odds/prob exist
    out["bet"] = (out["edge"] >= edge_threshold) & out[odds_col].notna() & out[prob_col].notna()

    # Profit calculation
    out["profit"] = 0.0
    win_mask = out["bet"] & (out[target_col] == 1)
    lose_mask = out["bet"] & (out[target_col] == 0)

    out.loc[win_mask, "profit"] = stake * (out.loc[win_mask, odds_col] - 1.0)
    out.loc[lose_mask, "profit"] = -stake

    # Running bankroll curve (starting at 0)
    out["cum_profit"] = out["profit"].cumsum()

    total_bets = int(out["bet"].sum())
    total_profit = float(out["profit"].sum())
    roi = float(total_profit / (total_bets * stake)) if total_bets > 0 else 0.0
    hit_rate = float(out.loc[out["bet"], target_col].mean()) if total_bets > 0 else 0.0

    # Max drawdown
    running_max = out["cum_profit"].cummax()
    drawdown = out["cum_profit"] - running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    summary = {
        "edge_threshold": edge_threshold,
        "stake": stake,
        "total_bets": total_bets,
        "hit_rate": round(hit_rate, 4),
        "total_profit": round(total_profit, 2),
        "roi": round(roi, 4),
        "max_drawdown": round(max_drawdown, 2),
    }

    return {"summary": summary, "results": out}


def pick_edge_threshold_by_roi(
    df: pd.DataFrame,
    prob_col: str,
    target_col: str = "target",
    odds_col: str = "odds",
    thresholds: list[float] | None = None,
    stake: float = 1.0,
) -> pd.DataFrame:
    """
    Quick sweep over edge_threshold values and return ROI table.
    """
    if thresholds is None:
        thresholds = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

    rows = []
    for t in thresholds:
        res = simulate_flat_betting(
            df=df,
            prob_col=prob_col,
            target_col=target_col,
            odds_col=odds_col,
            edge_threshold=t,
            stake=stake,
        )["summary"]
        rows.append(res)

    return pd.DataFrame(rows).sort_values(["roi", "total_profit"], ascending=False).reset_index(drop=True)