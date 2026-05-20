import json
from pathlib import Path

import requests
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

RAW_DATA_DIR = Path("data/raw")
BRASILEIRAO_SERIE_A = 71
BASE_URL = "https://v3.football.api-sports.io"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    api_football_key: str


def fetch_fixtures(season: int, settings: Settings) -> list[dict]:
    url = f"{BASE_URL}/fixtures"
    headers = {"x-apisports-key": settings.api_football_key}
    params = {"league": BRASILEIRAO_SERIE_A, "season": season}

    logger.info(f"Fetching fixtures: season={season}")
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    fixtures = response.json()["response"]
    logger.info(f"Fetched {len(fixtures)} fixtures for season {season}")
    return fixtures


def save_fixtures(fixtures: list[dict], season: int) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DATA_DIR / f"brasileirao_serie_a_{season}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved to {path}")
    return path
