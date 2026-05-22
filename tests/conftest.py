import pytest


@pytest.fixture
def raw_fixtures():
    """Minimal API-Football fixtures for two rounds."""

    def _make(fixture_id, date, home_id, away_id, home_goals, away_goals, round_n):
        return {
            "fixture": {
                "id": fixture_id,
                "date": date,
                "referee": "Ref",
                "venue": {"name": "Stadium"},
                "status": {"short": "FT"},
            },
            "league": {
                "id": 71,
                "season": 2024,
                "round": f"Regular Season - {round_n}",
            },
            "teams": {
                "home": {"id": home_id, "name": f"Team{home_id}"},
                "away": {"id": away_id, "name": f"Team{away_id}"},
            },
            "goals": {"home": home_goals, "away": away_goals},
            "score": {
                "halftime": {"home": 0, "away": 0},
                "fulltime": {"home": home_goals, "away": away_goals},
            },
        }

    return [
        _make(1, "2024-01-01T00:00:00+00:00", 1, 2, 2, 1, 1),  # H
        _make(2, "2024-01-01T00:00:00+00:00", 3, 4, 0, 0, 1),  # D
        _make(3, "2024-01-08T00:00:00+00:00", 2, 3, 1, 2, 2),  # A
        _make(4, "2024-01-08T00:00:00+00:00", 4, 1, 1, 3, 2),  # A
    ]


@pytest.fixture
def parsed_df(raw_fixtures):
    from chute_certo.ingestion.processing import parse_fixtures

    return parse_fixtures(raw_fixtures)
