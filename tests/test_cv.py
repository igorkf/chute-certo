from chute_certo.features.build import build_features
from chute_certo.training.cv import walk_forward_splits


def test_val_always_after_train(parsed_df):
    features = build_features(parsed_df)
    for train, val in walk_forward_splits(features, min_train_rounds=1):
        assert train["date"].max() < val["date"].min()


def test_min_train_rounds_respected(parsed_df):
    features = build_features(parsed_df)
    splits = list(walk_forward_splits(features, min_train_rounds=1))
    # with 2 rounds and min_train_rounds=1, should yield exactly 1 split
    assert len(splits) == 1


def test_train_grows_each_fold(parsed_df):
    features = build_features(parsed_df)
    splits = list(walk_forward_splits(features, min_train_rounds=1))
    train_sizes = [len(train) for train, _ in splits]
    assert train_sizes == sorted(train_sizes)
