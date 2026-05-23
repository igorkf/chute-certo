"""Predict the next upcoming round and save results to SQLite."""

import os
import sys

import dagshub
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from chute_certo.features.build import WINDOW, compute_current_form
from chute_certo.ingestion.api_football import Settings, fetch_fixtures
from chute_certo.ingestion.processing import load_seasons, parse_fixtures, parse_round
from chute_certo.predictions.store import save_predictions

HISTORICAL_SEASONS = [2023, 2024, 2025]

load_dotenv()
dagshub.init(
    repo_owner=os.environ["DAGSHUB_REPO_OWNER"],
    repo_name=os.environ["DAGSHUB_REPO_NAME"],
    mlflow=True,
)


def predict_upcoming(current_season: int) -> None:
    settings = Settings()

    logger.info(f"Fetching fixtures for season {current_season}...")
    raw = fetch_fixtures(current_season, settings)

    finished = [f for f in raw if f["fixture"]["status"]["short"] == "FT"]
    upcoming = [f for f in raw if f["fixture"]["status"]["short"] == "NS"]

    if not upcoming:
        logger.info("No upcoming fixtures found.")
        return

    by_round: dict[int, list] = {}
    for f in upcoming:
        r = parse_round(f["league"]["round"])
        by_round.setdefault(r, []).append(f)
    next_round = min(by_round)
    next_fixtures = by_round[next_round]
    logger.info(f"Predicting round {next_round}: {len(next_fixtures)} matches")

    historical_df = load_seasons(HISTORICAL_SEASONS)
    current_df = parse_fixtures(finished)
    context_df = (
        pd.concat([historical_df, current_df], ignore_index=True)
        if not current_df.empty
        else historical_df
    )

    model = mlflow.sklearn.load_model(os.environ["MODEL_URI"])

    rows = []
    for f in next_fixtures:
        home_id = f["teams"]["home"]["id"]
        away_id = f["teams"]["away"]["id"]

        form = compute_current_form(context_df, [home_id, away_id], window=WINDOW)

        if home_id not in form.index or away_id not in form.index:
            logger.warning(
                f"Missing form data for fixture {f['fixture']['id']}, skipping"
            )
            continue

        X = [
            [
                form.loc[home_id, "form_points"],
                form.loc[home_id, "form_scored"],
                form.loc[home_id, "form_conceded"],
                form.loc[away_id, "form_points"],
                form.loc[away_id, "form_scored"],
                form.loc[away_id, "form_conceded"],
            ]
        ]

        proba = model.predict_proba(X)[0]
        proba_map = dict(zip(model.classes_, proba, strict=True))
        predicted = max(proba_map, key=proba_map.get)

        rows.append(
            {
                "fixture_id": f["fixture"]["id"],
                "season": current_season,
                "round": next_round,
                "match_date": f["fixture"]["date"],
                "home_team_id": home_id,
                "home_team": f["teams"]["home"]["name"],
                "away_team_id": away_id,
                "away_team": f["teams"]["away"]["name"],
                "prob_home": round(proba_map.get("H", 0.0), 4),
                "prob_draw": round(proba_map.get("D", 0.0), 4),
                "prob_away": round(proba_map.get("A", 0.0), 4),
                "predicted_result": predicted,
            }
        )

    if rows:
        save_predictions(rows)
        logger.info(f"Saved {len(rows)} predictions for round {next_round}")
        for r in rows:
            logger.info(
                f"  {r['home_team']} vs {r['away_team']}: "
                f"H={r['prob_home']:.1%} D={r['prob_draw']:.1%} "
                f"A={r['prob_away']:.1%} → {r['predicted_result']}"
            )


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    predict_upcoming(season)
