"""Fetch actual results and update predictions in Turso."""

import sys

from dotenv import load_dotenv
from loguru import logger

from chute_certo.ingestion.api_football import Settings, fetch_fixtures
from chute_certo.ingestion.processing import result
from chute_certo.predictions.store import (
    batch_update_actual_results,
    load_pending_predictions,
)

load_dotenv()


def evaluate_round(current_season: int) -> None:
    settings = Settings()
    raw = fetch_fixtures(current_season, settings)

    finished = {
        f["fixture"]["id"]: result(f["goals"]["home"], f["goals"]["away"])
        for f in raw
        if f["fixture"]["status"]["short"] == "FT"
        and f["goals"]["home"] is not None
        and f["goals"]["away"] is not None
    }

    pending = load_pending_predictions()

    if not pending:
        logger.info("No pending predictions to evaluate.")
        return

    updates: list[tuple[int, str]] = []
    correct = 0
    for pred in pending:
        fid = pred["fixture_id"]
        if fid not in finished:
            continue
        actual = finished[fid]
        updates.append((fid, actual))
        hit = actual == pred["predicted_result"]
        correct += hit
        mark = "✓" if hit else "✗"
        logger.info(
            f"  {mark} {pred['home_team']} vs {pred['away_team']}: "
            f"predicted={pred['predicted_result']} actual={actual}"
        )

    if updates:
        batch_update_actual_results(updates)
        logger.info(
            f"Evaluated {len(updates)} predictions — {correct}/{len(updates)} correct "
            f"({correct / len(updates):.0%})"
        )


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    evaluate_round(season)
