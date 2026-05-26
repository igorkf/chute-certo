import os
from datetime import UTC, datetime

import libsql_client
from dotenv import load_dotenv

load_dotenv()

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS predictions (
        fixture_id       INTEGER PRIMARY KEY,
        season           INTEGER,
        round            INTEGER,
        match_date       TEXT,
        home_team_id     INTEGER,
        home_team        TEXT,
        away_team_id     INTEGER,
        away_team        TEXT,
        prob_home        REAL,
        prob_draw        REAL,
        prob_away        REAL,
        predicted_result TEXT,
        actual_result    TEXT,
        created_at       TEXT
    )
"""


def _client() -> libsql_client.ClientSync:
    url = os.environ["TURSO_URL"].replace("libsql://", "https://")
    return libsql_client.create_client_sync(
        url=url,
        auth_token=os.environ["TURSO_AUTH_TOKEN"],
    )


def save_predictions(rows: list[dict]) -> None:
    now = datetime.now(UTC).isoformat()
    with _client() as client:
        client.execute(_CREATE_TABLE)
        client.batch(
            [
                libsql_client.Statement(
                    """
                INSERT OR REPLACE INTO predictions
                    (fixture_id, season, round, match_date,
                     home_team_id, home_team, away_team_id, away_team,
                     prob_home, prob_draw, prob_away, predicted_result, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        row["fixture_id"],
                        row["season"],
                        row["round"],
                        row["match_date"],
                        row["home_team_id"],
                        row["home_team"],
                        row["away_team_id"],
                        row["away_team"],
                        row["prob_home"],
                        row["prob_draw"],
                        row["prob_away"],
                        row["predicted_result"],
                        now,
                    ],
                )
                for row in rows
            ]
        )


def load_predictions() -> list[dict]:
    with _client() as client:
        client.execute(_CREATE_TABLE)
        rs = client.execute("SELECT * FROM predictions ORDER BY match_date")
    return [dict(zip(rs.columns, row, strict=True)) for row in rs.rows]


def update_actual_result(fixture_id: int, actual_result: str) -> None:
    with _client() as client:
        client.execute(
            libsql_client.Statement(
                "UPDATE predictions SET actual_result = ? WHERE fixture_id = ?",
                [actual_result, fixture_id],
            )
        )
