from collections.abc import Iterator

import pandas as pd


def walk_forward_splits(
    df: pd.DataFrame,
    min_train_rounds: int = 10,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Yields (train, val) pairs for walk-forward validation by round.

    For each round R, train contains all games from rounds before R
    and val contains the games in round R. Starts after min_train_rounds
    to ensure the model has enough history before the first prediction.

    Features built on the full dataset are safe to use here because
    rolling stats are computed with .shift(1), so round R features
    only depend on rounds < R (no leakage).
    """
    df = df.sort_values("date").copy()
    df["round_idx"] = df.groupby(["season", "round"], sort=False).ngroup()
    n_rounds = df["round_idx"].nunique()

    for i in range(min_train_rounds, n_rounds):
        train = df[df["round_idx"] < i].drop(columns="round_idx")
        val = df[df["round_idx"] == i].drop(columns="round_idx")
        yield train, val
