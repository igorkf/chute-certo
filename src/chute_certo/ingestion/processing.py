import json

import pandas as pd
from loguru import logger

from chute_certo.ingestion.api_football import RAW_DATA_DIR


def parse_round(round_str: str) -> int:
    return int(round_str.split(" - ")[-1])


def _result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def parse_fixtures(fixtures: list[dict]) -> pd.DataFrame:
    rows = []
    for f in fixtures:
        if f["fixture"]["status"]["short"] != "FT":
            continue

        home_goals = f["goals"]["home"]
        away_goals = f["goals"]["away"]

        rows.append(
            {
                "fixture_id": f["fixture"]["id"],
                "date": f["fixture"]["date"],
                "season": f["league"]["season"],
                "round": parse_round(f["league"]["round"]),
                "home_team_id": f["teams"]["home"]["id"],
                "home_team": f["teams"]["home"]["name"],
                "away_team_id": f["teams"]["away"]["id"],
                "away_team": f["teams"]["away"]["name"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "ht_home_goals": f["score"]["halftime"]["home"],
                "ht_away_goals": f["score"]["halftime"]["away"],
                "venue": f["fixture"]["venue"]["name"],
                "referee": f["fixture"]["referee"],
                "result": _result(home_goals, away_goals),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"Parsed {len(df)} finished fixtures")
    return df


def load_seasons(seasons: list[int]) -> pd.DataFrame:
    dfs = []
    for season in seasons:
        path = RAW_DATA_DIR / f"brasileirao_serie_a_{season}.json"
        if not path.exists():
            logger.warning(f"Season {season} not found — run download_data.py first")
            continue
        with open(path, encoding="utf-8") as f:
            fixtures = json.load(f)
        dfs.append(parse_fixtures(fixtures))

    if not dfs:
        raise FileNotFoundError(
            "No season data found. Run: uv run python scripts/download_data.py"
        )

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} total fixtures across {len(dfs)} season(s)")
    return df
