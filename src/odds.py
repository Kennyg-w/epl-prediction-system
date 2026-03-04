# src/odds.py
from __future__ import annotations

import pandas as pd


def standardize_team_names(s: pd.Series) -> pd.Series:
    """Basic cleanup hook (you can expand mappings later)."""
    return s.astype(str).str.strip()


def merge_odds(
    matches: pd.DataFrame,
    odds: pd.DataFrame,
    matches_date_col: str = "date",
    matches_team_col: str = "team",
    matches_opp_col: str = "opponent",
    odds_date_col: str = "date",
    odds_home_col: str = "HomeTeam",
    odds_away_col: str = "AwayTeam",
    odds_home_odds_col: str = "B365H",
    # We will store the odds for "team to win" as `odds`
) -> pd.DataFrame:
    """
    Merge bookmaker odds into a per-team-per-match dataset.

    Assumes:
    - `matches` has one row per team per match (team vs opponent)
    - `odds` has one row per match with home/away teams and home win odds.

    Logic:
    - If matches.venue == 'Home' then team==HomeTeam -> odds = home odds
    - If matches.venue == 'Away' then team==AwayTeam -> odds = away odds (if you provide it)

    NOTE: This function uses ONLY home odds column by default.
    If your odds file also has away odds (e.g., B365A), pass it and expand logic.
    """
    m = matches.copy()
    o = odds.copy()

    m[matches_date_col] = pd.to_datetime(m[matches_date_col])
    o[odds_date_col] = pd.to_datetime(o[odds_date_col])

    m[matches_team_col] = standardize_team_names(m[matches_team_col])
    m[matches_opp_col] = standardize_team_names(m[matches_opp_col])

    o[odds_home_col] = standardize_team_names(o[odds_home_col])
    o[odds_away_col] = standardize_team_names(o[odds_away_col])

    merged = m.merge(
        o[[odds_date_col, odds_home_col, odds_away_col, odds_home_odds_col]],
        left_on=[matches_date_col, matches_team_col, matches_opp_col],
        right_on=[odds_date_col, odds_home_col, odds_away_col],
        how="left",
    )

    merged = merged.rename(columns={odds_home_odds_col: "odds"})

    # drop join helper cols
    merged = merged.drop(columns=[odds_date_col, odds_home_col, odds_away_col], errors="ignore")

    return merged