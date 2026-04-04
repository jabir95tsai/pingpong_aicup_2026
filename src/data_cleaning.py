"""Data cleaning: 7-step preprocessing pipeline."""
import pandas as pd
import numpy as np


# Step 1: strikeId mapping {1,2,4,8,16} -> {0,1,2,3,4}
STRIKE_ID_MAP = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}


def clean_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Apply all 7 cleaning steps. Returns cleaned copies."""
    train = train_df.copy()
    test = test_df.copy()

    # Step 1: strikeId remap to continuous indices
    train["strikeId"] = train["strikeId"].map(STRIKE_ID_MAP).fillna(0).astype(int)
    test["strikeId"] = test["strikeId"].map(STRIKE_ID_MAP).fillna(0).astype(int)

    # Step 2: Player ID remap to continuous 0..N (train+test combined)
    all_players = set(train["gamePlayerId"].unique()) | set(train["gamePlayerOtherId"].unique()) \
                | set(test["gamePlayerId"].unique()) | set(test["gamePlayerOtherId"].unique())
    player_map = {pid: idx for idx, pid in enumerate(sorted(all_players))}
    for col in ["gamePlayerId", "gamePlayerOtherId"]:
        train[col] = train[col].map(player_map)
        test[col] = test[col].map(player_map)

    # Step 3: numberGame clip to 7 (max 7 games in table tennis)
    train["numberGame"] = train["numberGame"].clip(upper=7)
    test["numberGame"] = test["numberGame"].clip(upper=7)

    # Step 4: serverGetPoint excluded from model input (handled in features)
    # (We keep the column for target extraction but don't use it as feature)

    # Step 5: Data validation - check no nulls, values in range
    for df, name in [(train, "train"), (test, "test")]:
        assert df.isnull().sum().sum() == 0, f"{name} has null values"
        assert df["handId"].between(0, 2).all(), f"{name} handId out of range"
        assert df["strengthId"].between(0, 3).all(), f"{name} strengthId out of range"
        assert df["spinId"].between(0, 5).all(), f"{name} spinId out of range"
        assert df["pointId"].between(0, 9).all(), f"{name} pointId out of range"
        assert df["actionId"].between(0, 18).all(), f"{name} actionId out of range"
        assert df["positionId"].between(0, 3).all(), f"{name} positionId out of range"

    # Step 6: Training data expansion (handled in features.py - every strike as target)

    # Step 7: Rally-based fold splitting (handled in training code)

    n_players = len(player_map)
    print(f"  strikeId remapped: {STRIKE_ID_MAP}")
    print(f"  Player IDs remapped: {n_players} unique players -> 0..{n_players-1}")
    print(f"  numberGame clipped to [1, 7]")
    print(f"  Data validation passed")

    return train, test, player_map
