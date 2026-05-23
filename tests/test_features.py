import pytest

from chute_certo.features.build import build_features, compute_current_form


def test_no_leakage_in_round_1(parsed_df):
    features = build_features(parsed_df)
    round1 = features[features["round"] == 1]
    assert round1["home_form_points"].isna().all()
    assert round1["away_form_points"].isna().all()


def test_features_populated_from_round_2(parsed_df):
    features = build_features(parsed_df)
    round2 = features[features["round"] == 2]
    assert round2["home_form_points"].notna().all()
    assert round2["away_form_points"].notna().all()


def test_feature_columns_present(parsed_df):
    features = build_features(parsed_df)
    expected = [
        "home_form_points",
        "home_form_scored",
        "home_form_conceded",
        "away_form_points",
        "away_form_scored",
        "away_form_conceded",
    ]
    for col in expected:
        assert col in features.columns


def test_row_count_preserved(parsed_df):
    features = build_features(parsed_df)
    assert len(features) == len(parsed_df)


def test_compute_current_form_all_teams_present(parsed_df):
    form = compute_current_form(parsed_df, [1, 2, 3, 4])
    assert set(form.index) == {1, 2, 3, 4}
    assert {"form_points", "form_scored", "form_conceded"}.issubset(form.columns)


def test_compute_current_form_includes_last_match(parsed_df):
    # Team 1 won both matches (3 pts each) — form should reflect the last match
    form = compute_current_form(parsed_df, [1])
    assert form.loc[1, "form_points"] == pytest.approx(3.0)


def test_compute_current_form_missing_team_excluded(parsed_df):
    form = compute_current_form(parsed_df, [1, 99])
    assert 1 in form.index
    assert 99 not in form.index
