from chute_certo.features.build import build_features


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
