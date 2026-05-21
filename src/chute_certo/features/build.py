import pandas as pd

WINDOW = 5


def _team_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape matches into one row per team per game."""
    cols = ["fixture_id", "date", "home_team_id", "home_goals", "away_goals", "result"]
    home = df[cols].copy()
    home = home.rename(
        columns={
            "home_team_id": "team_id",
            "home_goals": "scored",
            "away_goals": "conceded",
        }
    )
    home["points"] = home["result"].map({"H": 3, "D": 1, "A": 0})

    cols = ["fixture_id", "date", "away_team_id", "away_goals", "home_goals", "result"]
    away = df[cols].copy()
    away = away.rename(
        columns={
            "away_team_id": "team_id",
            "away_goals": "scored",
            "home_goals": "conceded",
        }
    )
    away["points"] = away["result"].map({"A": 3, "D": 1, "H": 0})

    return pd.concat([home, away], ignore_index=True).sort_values(["team_id", "date"])


def _rolling_team_stats(team_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute rolling stats per team using only past games (.shift(1) prevents leakage).
    Returns one row per fixture_id with prefixed columns (home_ or away_).
    """
    cols = ["scored", "conceded", "points"]
    rolled = team_df.groupby("team_id")[cols].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    team_df = team_df.copy()
    team_df["form_scored"] = rolled["scored"]
    team_df["form_conceded"] = rolled["conceded"]
    team_df["form_points"] = rolled["points"]
    keep = ["fixture_id", "team_id", "form_scored", "form_conceded", "form_points"]
    return team_df[keep]


def build_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """
    Add rolling form features for home and away teams.
    Only data from matches strictly before each game is used.
    """
    team_df = _team_perspective(df)
    stats = _rolling_team_stats(team_df, window)

    home_stats = stats.merge(
        df[["fixture_id", "home_team_id"]],
        on="fixture_id",
    )
    home_stats = home_stats[home_stats["team_id"] == home_stats["home_team_id"]].drop(
        columns=["team_id", "home_team_id"]
    )
    home_stats.columns = [
        "fixture_id" if c == "fixture_id" else f"home_{c}" for c in home_stats.columns
    ]

    away_stats = stats.merge(
        df[["fixture_id", "away_team_id"]],
        on="fixture_id",
    )
    away_stats = away_stats[away_stats["team_id"] == away_stats["away_team_id"]].drop(
        columns=["team_id", "away_team_id"]
    )
    away_stats.columns = [
        "fixture_id" if c == "fixture_id" else f"away_{c}" for c in away_stats.columns
    ]

    result = df.merge(home_stats, on="fixture_id").merge(away_stats, on="fixture_id")
    return result
