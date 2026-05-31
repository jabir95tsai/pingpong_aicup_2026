"""v15feat_e_nomismatch — ablation of v15feat_e (R-070 ablation per Codex 2026-05-24).

Codex verdict on the original 7-feature R-070 smoke: SN-aware profile is mixed
(SN≤2 hurts, SN 3-4 lifts, SN≥5 hurts). Codex hypothesised the SN≤2 regression
came from the `stroke_position_mismatch_proxy` flag (which fires heavily on
serves where positionId encoding is dominated by serve-side conventions).

This ablation drops:
  - stroke_position_mismatch_proxy
  - mismatch_AND_far_gap  (interaction that depends on the above)

Keeps the 5 PURE PROBABILITY/MISSINGNESS/GAP features:
  1. last_point_side       — 0=missing, 1=FH(1,4,7), 2=mid(2,5,8), 3=BH(3,6,9)
  2. last_point_depth      — 0=missing, 1=short(1-3), 2=half(4-6), 3=long(7-9)
  3. last_point_valid      — missingness flag (pointId > 0)
  4. last_position_valid   — missingness flag (positionId > 0)
  5. last_outgoing_lateral_gap — abs(positionId - last_point_side) when both valid; else 0

Per Codex: tests + Fold-1 smoke only. No Group C extension. No full 5-fold
until Codex reviews this ablation artifact.
"""
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_v15feat import (  # noqa: E402
    build_features_v15feat,
    compute_global_stats_v15feat,
    get_feature_names_v15feat,
)


def compute_global_stats_v15feat_e_nomismatch(train_df: pd.DataFrame) -> dict:
    """Same as v15feat (no new global stats)."""
    return compute_global_stats_v15feat(train_df)


def get_feature_names_v15feat_e_nomismatch(feat_df: pd.DataFrame) -> list:
    return get_feature_names_v15feat(feat_df)


def _point_to_side(p: int) -> int:
    if p == 0:
        return 0
    return ((p - 1) % 3) + 1


def _point_to_depth(p: int) -> int:
    if p == 0:
        return 0
    return ((p - 1) // 3) + 1


def build_features_v15feat_e_nomismatch(
    df: pd.DataFrame,
    is_train: bool,
    global_stats_v9: dict,
    raw_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build v15feat features + 5 v15feat_e_nomismatch features.

    Same prefix-only construction as v15feat_e. Drops only the mismatch proxy
    and its interaction. NEVER reads serverGetPoint.
    """
    feat_df = build_features_v15feat(
        df, is_train=is_train,
        global_stats_v9=global_stats_v9,
        raw_df=raw_df,
    )

    if raw_df is None:
        raw_df = df

    rally_cache: dict[int, dict] = {}
    raw_sorted = raw_df.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
    for rid, grp in raw_sorted.groupby("rally_uid", sort=False):
        rally_cache[int(rid)] = {
            "strike": grp["strikeNumber"].values.astype(np.int32),
            "pos":    grp["positionId"].values.astype(np.int32),
            "pt":     grp["pointId"].values.astype(np.int32),
        }

    n_rows = len(feat_df)
    rid_arr = feat_df["rally_uid"].astype(np.int64).values
    nsn_arr = feat_df["next_strikeNumber"].astype(np.int32).values

    out_point_side = np.zeros(n_rows, dtype=np.float32)
    out_point_depth = np.zeros(n_rows, dtype=np.float32)
    out_point_valid = np.zeros(n_rows, dtype=np.float32)
    out_position_valid = np.zeros(n_rows, dtype=np.float32)
    out_lateral_gap = np.zeros(n_rows, dtype=np.float32)

    for i in range(n_rows):
        rid = int(rid_arr[i])
        next_sn = int(nsn_arr[i])
        cache = rally_cache.get(rid)
        if cache is None or next_sn <= 1:
            continue
        prefix_mask = cache["strike"] < next_sn
        if not prefix_mask.any():
            continue
        prefix_strike = cache["strike"][prefix_mask]
        last_idx = int(np.argmax(prefix_strike))
        last_pos = int(cache["pos"][prefix_mask][last_idx])
        last_pt = int(cache["pt"][prefix_mask][last_idx])

        out_point_side[i] = float(_point_to_side(last_pt))
        out_point_depth[i] = float(_point_to_depth(last_pt))
        out_point_valid[i] = 1.0 if last_pt > 0 else 0.0
        out_position_valid[i] = 1.0 if last_pos > 0 else 0.0
        if last_pos > 0 and out_point_side[i] > 0:
            out_lateral_gap[i] = float(abs(last_pos - int(out_point_side[i])))

    feat_df["last_point_side"] = out_point_side
    feat_df["last_point_depth"] = out_point_depth
    feat_df["last_point_valid"] = out_point_valid
    feat_df["last_position_valid"] = out_position_valid
    feat_df["last_outgoing_lateral_gap"] = out_lateral_gap

    return feat_df


V15FEAT_E_NOMISMATCH_ADDED_COLUMNS = [
    "last_point_side",
    "last_point_depth",
    "last_point_valid",
    "last_position_valid",
    "last_outgoing_lateral_gap",
]

assert len(V15FEAT_E_NOMISMATCH_ADDED_COLUMNS) == 5, (
    f"v15feat_e_nomismatch column count drift: {len(V15FEAT_E_NOMISMATCH_ADDED_COLUMNS)} != 5"
)
