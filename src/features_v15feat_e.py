"""Feature engineering v15feat_e — neutral stroke-position core (R-070).

Per Codex `APPROVE_WITH_FIXES` 2026-05-24 (6 fixes applied):

  1. Do NOT describe `handId==1 AND positionId==1` as cross-body reach.
     handId is FH/BH stroke type, NOT physical handedness. Use neutral
     `stroke_position_mismatch_proxy` framing.
  2. Drop raw `position_change_in_prefix` (consecutive shots alternate
     hitters → not same-player movement).
  3. Replace 1D distance with 2D (side, depth) decomposition for pointId.
  4. Ship a small core (6-7 features). Defer Group C inflation.
  5. Missingness flags for `positionId=0` / `pointId=0` (68.82% sparse).
  6. Smoke report must include coverage + per-SN slices.

User intuition (2026-05-24): "if he's a right hand and he uses maybe right hand
on the left the next ball if its on the right is harder" → encoded as neutral
stroke-position pair indicators + outgoing-lateral-gap, not as physical
handedness claims.

Feature additions on top of v15feat_a (7 features, all prefix-only):

  Group A — Neutral stroke-position pair flag (1 feature)
    stroke_position_mismatch_proxy : 1 if last shot's (handId, positionId) ∈
      {(1, 1), (2, 3)} — FH-stroke-on-left OR BH-stroke-on-right. Codex-neutral
      interpretation: "stroke type opposite to nominal court-side" without
      claiming physical-hand semantics.

  Group B — 2D pointId decomposition (4 features, with missingness)
    last_point_side    : 0=missing(pointId==0), 1=FH-side(1,4,7), 2=mid(2,5,8), 3=BH-side(3,6,9)
    last_point_depth   : 0=missing, 1=short(1-3), 2=half(4-6), 3=long(7-9)
    last_point_valid   : 1 if pointId > 0 else 0 (missingness flag)
    last_position_valid: 1 if positionId > 0 else 0 (missingness flag)

  Group C — Outgoing lateral gap (1 feature)
    last_outgoing_lateral_gap : abs(positionId - last_point_side) when both
      valid; else 0. (Higher = larger lateral movement implied for the
      receiver to reach the next ball.)

  OPTIONAL interaction (1 feature, included per Codex "at most one interaction"):
    mismatch_AND_far_gap : stroke_position_mismatch_proxy AND
      (last_outgoing_lateral_gap >= 2)

Total: 7 features.

Empty-prefix / missing-data default: all features = 0.

Per Codex: tests must verify prefix-only construction, no SGP read,
exact feature count, NaN/Inf absence, and missingness flag honesty.
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

# Neutral framing: these are stroke-position pairs that are "off-natural-side",
# NOT cross-body reach (which would require knowing physical handedness).
MISMATCH_PAIRS = frozenset({(1, 1), (2, 3)})  # (handId, positionId)


def compute_global_stats_v15feat_e(train_df: pd.DataFrame) -> dict:
    """No new global stats needed (all features are per-rally prefix observable)."""
    return compute_global_stats_v15feat(train_df)


def get_feature_names_v15feat_e(feat_df: pd.DataFrame) -> list:
    return get_feature_names_v15feat(feat_df)


def _point_to_side(p: int) -> int:
    """pointId → side: 0=missing, 1=FH(1,4,7), 2=mid(2,5,8), 3=BH(3,6,9)."""
    if p == 0:
        return 0
    return ((p - 1) % 3) + 1


def _point_to_depth(p: int) -> int:
    """pointId → depth: 0=missing, 1=short(1-3), 2=half(4-6), 3=long(7-9)."""
    if p == 0:
        return 0
    return ((p - 1) // 3) + 1


def build_features_v15feat_e(
    df: pd.DataFrame,
    is_train: bool,
    global_stats_v9: dict,
    raw_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build v15feat features + 7 v15feat_e movement/position features.

    Per Codex fixes: neutral framing, 2D pointId decomposition with
    missingness flags, no player-aggregate features.

    NEVER reads `serverGetPoint`. All inputs are observable prefix shots:
    handId, positionId, pointId only.
    """
    feat_df = build_features_v15feat(
        df, is_train=is_train,
        global_stats_v9=global_stats_v9,
        raw_df=raw_df,
    )

    if raw_df is None:
        raw_df = df

    # Cache per-rally arrays for prefix lookup
    rally_cache: dict[int, dict] = {}
    raw_sorted = raw_df.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
    for rid, grp in raw_sorted.groupby("rally_uid", sort=False):
        rally_cache[int(rid)] = {
            "strike":   grp["strikeNumber"].values.astype(np.int32),
            "hand":     grp["handId"].values.astype(np.int32),
            "pos":      grp["positionId"].values.astype(np.int32),
            "pt":       grp["pointId"].values.astype(np.int32),
        }

    n_rows = len(feat_df)
    rid_arr = feat_df["rally_uid"].astype(np.int64).values
    nsn_arr = feat_df["next_strikeNumber"].astype(np.int32).values

    # Output buffers — exactly 7 features
    out_mismatch_proxy = np.zeros(n_rows, dtype=np.float32)
    out_point_side = np.zeros(n_rows, dtype=np.float32)
    out_point_depth = np.zeros(n_rows, dtype=np.float32)
    out_point_valid = np.zeros(n_rows, dtype=np.float32)
    out_position_valid = np.zeros(n_rows, dtype=np.float32)
    out_lateral_gap = np.zeros(n_rows, dtype=np.float32)
    out_mismatch_and_far = np.zeros(n_rows, dtype=np.float32)

    for i in range(n_rows):
        rid = int(rid_arr[i])
        next_sn = int(nsn_arr[i])
        cache = rally_cache.get(rid)
        if cache is None or next_sn <= 1:
            # next_sn=1 → predicting the serve; no prefix to read
            continue
        prefix_mask = cache["strike"] < next_sn
        if not prefix_mask.any():
            continue

        # LAST shot in prefix (strictly before target)
        prefix_strike = cache["strike"][prefix_mask]
        last_idx_in_prefix = int(np.argmax(prefix_strike))  # = arg of largest strike
        last_hand = int(cache["hand"][prefix_mask][last_idx_in_prefix])
        last_pos = int(cache["pos"][prefix_mask][last_idx_in_prefix])
        last_pt = int(cache["pt"][prefix_mask][last_idx_in_prefix])

        # Group A — neutral mismatch proxy
        if (last_hand, last_pos) in MISMATCH_PAIRS:
            out_mismatch_proxy[i] = 1.0

        # Group B — 2D pointId decomposition + missingness flags
        out_point_side[i] = float(_point_to_side(last_pt))
        out_point_depth[i] = float(_point_to_depth(last_pt))
        out_point_valid[i] = 1.0 if last_pt > 0 else 0.0
        out_position_valid[i] = 1.0 if last_pos > 0 else 0.0

        # Group C — lateral gap (only when both pos and point side are valid)
        if last_pos > 0 and out_point_side[i] > 0:
            out_lateral_gap[i] = float(abs(last_pos - int(out_point_side[i])))

        # Optional interaction
        if out_mismatch_proxy[i] > 0 and out_lateral_gap[i] >= 2.0:
            out_mismatch_and_far[i] = 1.0

    # Materialise columns
    feat_df["stroke_position_mismatch_proxy"] = out_mismatch_proxy
    feat_df["last_point_side"] = out_point_side
    feat_df["last_point_depth"] = out_point_depth
    feat_df["last_point_valid"] = out_point_valid
    feat_df["last_position_valid"] = out_position_valid
    feat_df["last_outgoing_lateral_gap"] = out_lateral_gap
    feat_df["mismatch_AND_far_gap"] = out_mismatch_and_far

    return feat_df


V15FEAT_E_ADDED_COLUMNS = [
    "stroke_position_mismatch_proxy",
    "last_point_side",
    "last_point_depth",
    "last_point_valid",
    "last_position_valid",
    "last_outgoing_lateral_gap",
    "mismatch_AND_far_gap",
]

assert len(V15FEAT_E_ADDED_COLUMNS) == 7, (
    f"v15feat_e column count drift: {len(V15FEAT_E_ADDED_COLUMNS)} != 7"
)
