import sqlite3
from datetime import UTC, datetime
from pathlib import Path

DB_PATH = Path("data/predictions.db")


def save_predictions(rows: list[dict]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                fixture_id     INTEGER PRIMARY KEY,
                season         INTEGER,
                round          INTEGER,
                match_date     TEXT,
                home_team_id   INTEGER,
                home_team      TEXT,
                away_team_id   INTEGER,
                away_team      TEXT,
                prob_home      REAL,
                prob_draw      REAL,
                prob_away      REAL,
                predicted_result TEXT,
                actual_result    TEXT,
                created_at       TEXT
            )
        """)
        now = datetime.now(UTC).isoformat()
        conn.executemany(
            """
            INSERT OR REPLACE INTO predictions
                (fixture_id, season, round, match_date,
                 home_team_id, home_team, away_team_id, away_team,
                 prob_home, prob_draw, prob_away, predicted_result, created_at)
            VALUES
                (:fixture_id, :season, :round, :match_date,
                 :home_team_id, :home_team, :away_team_id, :away_team,
                 :prob_home, :prob_draw, :prob_away, :predicted_result, :created_at)
            """,
            [{**row, "created_at": now} for row in rows],
        )


def load_predictions() -> list[dict]:
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM predictions ORDER BY match_date").fetchall()
    return [dict(r) for r in rows]


def update_actual_result(fixture_id: int, actual_result: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE predictions SET actual_result = ? WHERE fixture_id = ?",
            (actual_result, fixture_id),
        )
