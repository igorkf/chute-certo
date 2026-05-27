"""Predict the next upcoming round and save results to Turso."""

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
from chute_certo.predictions.store import has_predictions_for_round, save_predictions

HISTORICAL_SEASONS = [2023, 2024, 2025]

load_dotenv()
dagshub.init(
    repo_owner=os.environ["DAGSHUB_REPO_OWNER"],
    repo_name=os.environ["DAGSHUB_REPO_NAME"],
    mlflow=True,
)


def _find_next_round(raw: list[dict]) -> tuple[int, list] | None:
    upcoming = [f for f in raw if f["fixture"]["status"]["short"] == "NS"]
    if not upcoming:
        return None
    by_round: dict[int, list] = {}
    for f in upcoming:
        r = parse_round(f["league"]["round"])
        by_round.setdefault(r, []).append(f)
    n = min(by_round)
    return n, by_round[n]


def predict_upcoming(current_season: int) -> None:
    settings = Settings()

    logger.info(f"Fetching fixtures for season {current_season}...")
    raw = fetch_fixtures(current_season, settings)

    finished = [f for f in raw if f["fixture"]["status"]["short"] == "FT"]
    found = _find_next_round(raw)

    if found is None:
        logger.info("No upcoming fixtures found.")
        return

    next_round, next_fixtures = found

    if has_predictions_for_round(current_season, next_round):
        logger.info(f"Round {next_round} already predicted, skipping.")
        return

    logger.info(f"Predicting round {next_round}: {len(next_fixtures)} matches")

    historical_df = load_seasons(HISTORICAL_SEASONS)
    current_df = parse_fixtures(finished)
    context_df = (
        pd.concat([historical_df, current_df], ignore_index=True)
        if not current_df.empty
        else historical_df
    )

    model = mlflow.sklearn.load_model(os.environ["MODEL_URI"])

    all_team_ids = list(
        {f["teams"]["home"]["id"] for f in next_fixtures}
        | {f["teams"]["away"]["id"] for f in next_fixtures}
    )
    form = compute_current_form(context_df, all_team_ids, window=WINDOW)

    rows = []
    for f in next_fixtures:
        home_id = f["teams"]["home"]["id"]
        away_id = f["teams"]["away"]["id"]

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("season", type=int, nargs="?", default=2026)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 0 if a new prediction is needed, 1 if already up to date.",
    )
    args = parser.parse_args()

    if args.check:
        raw = fetch_fixtures(args.season, Settings())
        found = _find_next_round(raw)
        if found is None:
            logger.info("No upcoming fixtures.")
            sys.exit(1)
        next_round, _ = found
        if has_predictions_for_round(args.season, next_round):
            logger.info(f"Round {next_round} already predicted.")
            sys.exit(1)
        logger.info(f"Round {next_round} needs prediction.")
        sys.exit(0)

    predict_upcoming(args.season)
