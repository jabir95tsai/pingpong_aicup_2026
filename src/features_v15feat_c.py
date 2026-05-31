"""R-047 features — v15feat_b + teammate v8's score-pressure features.

Builds on top of v15feat_b (which already includes v15feat aggregates +
transition priors). Adds 8 cheap score-state features inspired by the
teammate v8 audit:

  - is_serve_side       — odd next_strikeNumber = server's turn (R-029a
                           may already have similar; double-check)
  - is_deuce            — scoreSelf >= 10 AND scoreOther >= 10
  - match_point_self    — scoreSelf >= 10 AND lead >= 0
  - match_point_other   — scoreOther >= 10 AND lead <= 0
  - total_points        — scoreSelf + scoreOther
  - points_to_win_self  — max(0, 11 - scoreSelf)
  - points_to_win_other — max(0, 11 - scoreOther)
  - score_lead_abs      — abs(scoreSelf - scoreOther)

All derived from the CONTEXT shot's scoreSelf / scoreOther — already in
the v15feat_b output (since v9 backbone passes those through). Just
adds 8 derived columns. No extra raw-data join needed.

Banned: nothing new — same regimen as v15feat / v15feat_b.

INTERFACE matches v15feat_b for plug-in compat:
  compute_global_stats_v15feat_c(train_df) -> stats_dict
  build_features_v15feat_c(df, is_train, global_stats_v9, raw_df) -> feature_df
  get_feature_names_v15feat_c(feat_df) -> list[str]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from features_v15feat_b import (  # noqa: E402
    build_features_v15feat_b,
    compute_global_stats_v15feat_b,
    get_feature_names_v15feat_b,
)


V15FEAT_C_ADDED_COLUMNS = [
    "is_serve_side",
    "is_deuce",
    "match_point_self",
    "match_point_other",
    "total_points",
    "points_to_win_self",
    "points_to_win_other",
    "score_lead_abs",
]
assert len(V15FEAT_C_ADDED_COLUMNS) == 8


def _add_pressure_features(feat: pd.DataFrame) -> pd.DataFrame:
    """Add 8 score-pressure features to feat DataFrame.

    Assumes feat already has scoreSelf, scoreOther, next_strikeNumber columns
    (true for v9-family backbones).
    """
    if "scoreSelf" not in feat.columns or "scoreOther" not in feat.columns:
        raise ValueError(
            "feat missing scoreSelf/scoreOther columns required by v15feat_c. "
            f"Available: {list(feat.columns[:30])}")

    s_self = feat["scoreSelf"].fillna(0).astype(np.int32)
    s_other = feat["scoreOther"].fillna(0).astype(np.int32)

    cols = {}
    cols["total_points"] = (s_self + s_other).astype(np.float32)
    cols["is_deuce"] = ((s_self >= 10) & (s_other >= 10)).astype(np.int8)
    cols["match_point_self"] = (
        (s_self >= 10) & ((s_self - s_other) >= 0)
    ).astype(np.int8)
    cols["match_point_other"] = (
        (s_other >= 10) & ((s_other - s_self) >= 0)
    ).astype(np.int8)
    cols["points_to_win_self"] = (11 - s_self).clip(lower=0).astype(np.int8)
    cols["points_to_win_other"] = (11 - s_other).clip(lower=0).astype(np.int8)
    cols["score_lead_abs"] = (s_self - s_other).abs().astype(np.int8)

    if "next_strikeNumber" in feat.columns:
        next_sn = feat["next_strikeNumber"].fillna(1).astype(np.int32)
        cols["is_serve_side"] = (next_sn % 2 == 1).astype(np.int8)
    else:
        # Fallback: assume serve side
        cols["is_serve_side"] = np.ones(len(feat), dtype=np.int8)

    # Drop any cols whose name already exists in feat (avoids duplicate
    # column names which crash downstream get_feature_names_v6).
    existing = set(feat.columns)
    cols_safe = {k: v for k, v in cols.items() if k not in existing}
    skipped = set(cols.keys()) - set(cols_safe.keys())
    if skipped:
        print(f"  [v15feat_c] WARN: dropping {len(skipped)} cols already in feat: {sorted(skipped)}")
    new_cols = pd.DataFrame(cols_safe, index=feat.index)
    return pd.concat([feat, new_cols], axis=1)


def compute_global_stats_v15feat_c(train_df: pd.DataFrame) -> dict:
    """No extra stats needed — pressure features are pure row-level functions
    of already-present columns."""
    return compute_global_stats_v15feat_b(train_df)


def build_features_v15feat_c(
    df: pd.DataFrame,
    is_train: bool,
    global_stats_v9: dict = None,
    raw_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build v15feat_c features = v15feat_b + 8 score-pressure features."""
    feat = build_features_v15feat_b(
        df, is_train=is_train,
        global_stats_v9=global_stats_v9,
        raw_df=raw_df,
    )
    feat = _add_pressure_features(feat)
    return feat


def get_feature_names_v15feat_c(feat_df: pd.DataFrame) -> list:
    base = get_feature_names_v15feat_b(feat_df)
    extra = [c for c in V15FEAT_C_ADDED_COLUMNS if c in feat_df.columns]
    return base + extra
