"""Feature engineering V6: extends V5 with wider temporal context.

Key additions over V5 (931 features):
1. Extra one-hot lag features: lags 4, 6, 8, 10 (fill gap between 3 and 5, add deeper history)
   - Each lag adds 7 columns × cat sizes: handId(3)+strengthId(4)+spinId(6)+
     pointId(10)+actionId(19)+positionId(4)+strikeId(5) = 51 features
   - 4 new lags × 51 = 204 new features → total ~1135 features

These extra lags help the models see:
- lag4: the shot 3 positions before the predicted one
- lag6/8/10: rally history beyond the immediate 5-shot window

Cumulative context features (action/point histograms) are already in V3 base features.
"""
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_v5 import (
    build_features_v5, compute_global_stats_v5, get_feature_names_v5,
)

# ---------------------------------------------------------------------------
# Extra lag steps on top of V3/V5's [1, 2, 3, 5]
# ---------------------------------------------------------------------------
V6_EXTRA_LAGS = [4, 6, 8, 10]

# Category sizes (must match _CAT_SIZES in features_v3.py)
_CAT_SIZES = {
    "actionId":   19,
    "pointId":    10,
    "handId":      3,
    "strengthId":  4,
    "spinId":      6,
    "positionId":  4,
    "strikeId":    5,  # 0-4 after remap
}


def add_extra_lag_ohe(feat_df: pd.DataFrame, raw_df: pd.DataFrame,
                       extra_lags: list = V6_EXTRA_LAGS) -> pd.DataFrame:
    """Append extra one-hot lag features to feat_df.

    For each lag k in extra_lags and each row i in feat_df:
      - Find the shot in raw_df with (rally_uid == row.rally_uid AND
        strikeNumber == row.next_strikeNumber - k)
      - One-hot encode its categorical features
      - If the shot doesn't exist (short rally), all bits remain 0

    Parameters
    ----------
    feat_df : DataFrame already built by build_features_v5 (includes rally_uid,
              next_strikeNumber)
    raw_df  : The cleaned training or test DataFrame (after clean_data)
    extra_lags : list of int, lag steps to add
    """
    # Build shot lookup: (rally_uid, strikeNumber) -> categorical values
    shot_cols = ["rally_uid", "strikeNumber"] + list(_CAT_SIZES.keys())
    shot_lookup = raw_df[shot_cols].copy()
    shot_lookup["strikeNumber"] = shot_lookup["strikeNumber"].astype(int)

    out = feat_df.copy()

    for lag in extra_lags:
        target_sn = (feat_df["next_strikeNumber"].values.astype(int) - lag)

        merge_left = pd.DataFrame({
            "rally_uid":     feat_df["rally_uid"].values,
            "strikeNumber":  target_sn,
        })

        merged = merge_left.merge(shot_lookup, on=["rally_uid", "strikeNumber"],
                                  how="left")
        # merged rows with no match → NaN (shot doesn't exist)

        for col, n_cls in _CAT_SIZES.items():
            vals = merged[col].fillna(-1).astype(int).values
            for c in range(n_cls):
                col_name = f"oh_lag{lag}_{col}_{c}"
                out[col_name] = (vals == c).astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# Public API (mirrors features_v5 API)
# ---------------------------------------------------------------------------

def compute_global_stats_v6(train_df: pd.DataFrame) -> dict:
    """Compute V5 stats (re-exported for V6 use)."""
    return compute_global_stats_v5(train_df)


def build_features_v6(df: pd.DataFrame, is_train: bool,
                       global_stats_v6: dict,
                       raw_df: pd.DataFrame = None) -> pd.DataFrame:
    """Build full V6 feature set.

    Parameters
    ----------
    df            : cleaned DataFrame (train or test slice)
    is_train      : True for training data, False for test
    global_stats_v6 : returned by compute_global_stats_v6
    raw_df        : the same cleaned full DataFrame (for lag lookups).
                    If None, uses df (fine for test; for fold-level training
                    pass the fold training DataFrame)
    """
    feat_df = build_features_v5(df, is_train=is_train,
                                 global_stats_v5=global_stats_v6)
    if raw_df is None:
        raw_df = df
    feat_df = add_extra_lag_ohe(feat_df, raw_df, V6_EXTRA_LAGS)
    return feat_df


def get_feature_names_v6(feat_df: pd.DataFrame) -> list:
    """Return numeric feature column names (exclude metadata / target columns)."""
    exclude = {
        "rally_uid", "y_actionId", "y_pointId", "y_serverGetPoint",
        "next_strikeNumber",
    }
    ok_dtypes = {np.float32, np.float64, np.int32, np.int64, int, float}
    names = []
    for c in feat_df.columns:
        if c in exclude:
            continue
        if feat_df[c].dtype.type in ok_dtypes or pd.api.types.is_numeric_dtype(feat_df[c]):
            names.append(c)
    return names
