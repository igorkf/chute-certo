from chute_certo.ingestion.processing import parse_fixtures


def test_result_assignment(raw_fixtures):
    df = parse_fixtures(raw_fixtures)
    assert df.loc[df["fixture_id"] == 1, "result"].iloc[0] == "H"
    assert df.loc[df["fixture_id"] == 2, "result"].iloc[0] == "D"
    assert df.loc[df["fixture_id"] == 3, "result"].iloc[0] == "A"


def test_only_finished_matches_included():
    unfinished = [
        {
            "fixture": {
                "id": 99,
                "date": "2024-01-01T00:00:00+00:00",
                "referee": "R",
                "venue": {"name": "S"},
                "status": {"short": "NS"},
            },
            "league": {"id": 71, "season": 2024, "round": "Regular Season - 1"},
            "teams": {"home": {"id": 1, "name": "A"}, "away": {"id": 2, "name": "B"}},
            "goals": {"home": None, "away": None},
            "score": {
                "halftime": {"home": None, "away": None},
                "fulltime": {"home": None, "away": None},
            },
        }
    ]
    df = parse_fixtures(unfinished)
    assert len(df) == 0


def test_sorted_by_date(raw_fixtures):
    df = parse_fixtures(raw_fixtures)
    assert df["date"].is_monotonic_increasing


def test_round_parsed_correctly(raw_fixtures):
    df = parse_fixtures(raw_fixtures)
    assert set(df["round"].unique()) == {1, 2}
