from collections.abc import Iterator

import pandas as pd


def walk_forward_splits(
    df: pd.DataFrame,
    min_train_rounds: int = 10,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Yields (train, val) pairs for walk-forward validation by round.

    Val is defined by round number. Train is all matches with date
    strictly before the earliest match in the val round — this avoids
    leakage from postponed matches that are played weeks after the round
    nominally started (common in the Brasileirão).
    """
    df = df.sort_values("date").copy()
    df["round_idx"] = df.groupby(["season", "round"], sort=False).ngroup()
    n_rounds = df["round_idx"].nunique()

    for i in range(min_train_rounds, n_rounds):
        val = df[df["round_idx"] == i].drop(columns="round_idx")
        val_min_date = df[df["round_idx"] == i]["date"].min()
        train = df[df["date"] < val_min_date].drop(columns="round_idx")
        if train.empty:
            continue
        yield train, val
