import json
from pathlib import Path

import requests
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

RAW_DATA_DIR = Path("data/raw")
BRASILEIRAO_SERIE_A = "BSA"
BASE_URL = "https://api.football-data.org/v4"

# football-data.org status → internal status used by processing.py
_STATUS_MAP = {
    "FINISHED": "FT",
    "AWARDED": "FT",
    "SCHEDULED": "NS",
    "TIMED": "NS",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    api_key_football_data: str


def _normalize(match: dict, season: int) -> dict:
    """Convert football-data.org match object to our internal fixture format."""
    status = _STATUS_MAP.get(match["status"], match["status"])
    return {
        "fixture": {
            "id": match["id"],
            "date": match["utcDate"],
            "referee": None,
            "venue": {"name": None},
            "status": {"short": status},
        },
        "league": {
            "id": 2013,
            "season": season,
            "round": f"Regular Season - {match['matchday']}",
        },
        "teams": {
            "home": {"id": match["homeTeam"]["id"], "name": match["homeTeam"]["name"]},
            "away": {"id": match["awayTeam"]["id"], "name": match["awayTeam"]["name"]},
        },
        "goals": {
            "home": match["score"]["fullTime"]["home"],
            "away": match["score"]["fullTime"]["away"],
        },
        "score": {
            "halftime": {
                "home": match["score"]["halfTime"]["home"],
                "away": match["score"]["halfTime"]["away"],
            },
            "fulltime": {
                "home": match["score"]["fullTime"]["home"],
                "away": match["score"]["fullTime"]["away"],
            },
        },
    }


def fetch_fixtures(season: int, settings: Settings) -> list[dict]:
    url = f"{BASE_URL}/competitions/{BRASILEIRAO_SERIE_A}/matches"
    headers = {"X-Auth-Token": settings.api_key_football_data}
    params = {"season": season}

    logger.info(f"Fetching fixtures: season={season}")
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    matches = response.json()["matches"]
    logger.info(f"Fetched {len(matches)} fixtures for season {season}")
    return [_normalize(m, season) for m in matches]


def save_fixtures(fixtures: list[dict], season: int) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DATA_DIR / f"brasileirao_serie_a_{season}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved to {path}")
    return path
