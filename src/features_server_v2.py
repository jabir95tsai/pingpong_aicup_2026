"""features_server_v2 — v1 + last-K-shots one-hot features.

R-006 v2: extends features_server_v1 with explicit per-shot one-hot features
for the LAST K visible shots (K=3) of each rally prefix. The hypothesis is
that aggregate histograms throw away temporal information; explicit "shot at
N-1, N-2, N-3" features can recover the per-shot baseline AUC (~0.61) and
possibly exceed it.

All Codex R-006 constraints from v1 still apply:
- No `gamePlayerId` / `gamePlayerOtherId` as encoded inputs.
- Every feature derived from rows with `strikeNumber < next_strikeNumber`
  (prefix only). Asserted at build time.
- No n_shots-of-rally, terminal parity, or rally-suffix aggregates.
- The new last-K shot features are prefix shots (specifically positions
  N-1, N-2, N-3 within the prefix), NOT future or target rows.

NEW feature blocks (added on top of v1 features):
- shot_lag1_*: one-hot of (actionId, handId, spinId, strengthId, pointId)
  for shot at strikeNumber == N-1 (most recent visible shot).
- shot_lag2_*: same for strikeNumber == N-2 (or zero-vector if not present).
- shot_lag3_*: same for strikeNumber == N-3 (or zero-vector if not present).
- shot_lag1_is_server / lag2_is_server / lag3_is_server: binary indicating
  whether the lag shot was hit by the server (defined as shooter at strike 1).
"""
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features_server_v1 import (
    CATEGORICAL_HISTS, _empty_features, _compute_row_features,
    feature_names as v1_feature_names,
    count_only_features as v1_count_only_features,
)

LAG_CATEGORICAL = [
    ("actionId", 15),
    ("handId", 3),
    ("spinId", 6),
    ("strengthId", 4),
    ("pointId", 10),
]
LAG_K = 3  # number of recent shots to one-hot encode


def _lag_features_block(prefix_df: pd.DataFrame, server_id: int) -> dict:
    """One-hot features for the last K prefix shots (newest first)."""
    feats = {}
    if len(prefix_df) == 0:
        # Zero-pad
        for k in range(1, LAG_K + 1):
            for col, nbin in LAG_CATEGORICAL:
                for i in range(nbin):
                    feats[f"shot_lag{k}_{col}_p{i}"] = 0.0
            feats[f"shot_lag{k}_is_server"] = 0
            feats[f"shot_lag{k}_present"] = 0
        return feats

    # Sort by strikeNumber DESCENDING — newest first.
    sorted_pref = prefix_df.sort_values("strikeNumber", ascending=False).reset_index(drop=True)
    for k in range(1, LAG_K + 1):
        if k - 1 < len(sorted_pref):
            row = sorted_pref.iloc[k - 1]
            for col, nbin in LAG_CATEGORICAL:
                hist = np.zeros(nbin, dtype=np.float32)
                v = int(row[col])
                if 0 <= v < nbin:
                    hist[v] = 1.0
                for i in range(nbin):
                    feats[f"shot_lag{k}_{col}_p{i}"] = float(hist[i])
            feats[f"shot_lag{k}_is_server"] = int(int(row["gamePlayerId"]) == server_id)
            feats[f"shot_lag{k}_present"] = 1
        else:
            for col, nbin in LAG_CATEGORICAL:
                for i in range(nbin):
                    feats[f"shot_lag{k}_{col}_p{i}"] = 0.0
            feats[f"shot_lag{k}_is_server"] = 0
            feats[f"shot_lag{k}_present"] = 0
    return feats


def _compute_v2_row_features(prefix_df: pd.DataFrame) -> dict:
    """v1 features + lag block."""
    feats = _compute_row_features(prefix_df)

    if len(prefix_df) == 0:
        server_id = -1
    else:
        strike1 = prefix_df[prefix_df["strikeNumber"] == 1]
        if len(strike1) > 0:
            server_id = int(strike1["gamePlayerId"].iloc[0])
        else:
            server_id = int(prefix_df.iloc[0]["gamePlayerId"])

    feats.update(_lag_features_block(prefix_df, server_id))
    return feats


def feature_names() -> list:
    cols = list(v1_feature_names())
    for k in range(1, LAG_K + 1):
        for col, nbin in LAG_CATEGORICAL:
            for i in range(nbin):
                cols.append(f"shot_lag{k}_{col}_p{i}")
        cols.append(f"shot_lag{k}_is_server")
        cols.append(f"shot_lag{k}_present")
    return cols


def count_only_features() -> list:
    """Same diagnostic subset; lag features add no count info."""
    return v1_count_only_features() + [f"shot_lag{k}_present" for k in range(1, LAG_K + 1)]


def build_features_server_v2(target_rows: pd.DataFrame,
                              raw_df: pd.DataFrame,
                              is_train: bool = True) -> pd.DataFrame:
    """Per-row prefix-only feature build (v1 + last-K shots one-hot)."""
    cols = ["rally_uid", "strikeNumber", "gamePlayerId", "actionId", "pointId",
            "handId", "spinId", "strengthId", "scoreSelf", "scoreOther",
            "numberGame", "sex"]
    raw = raw_df[cols].copy()
    raw["strikeNumber"] = raw["strikeNumber"].astype(int)
    raw["gamePlayerId"] = raw["gamePlayerId"].astype(int)
    rally_to_rows = {rid: g.sort_values("strikeNumber").reset_index(drop=True)
                     for rid, g in raw.groupby("rally_uid", sort=False)}

    rally_uids = target_rows["rally_uid"].to_numpy()
    next_sns = target_rows["next_strikeNumber"].to_numpy(dtype=int)

    out_rows = []
    max_src_violations = 0
    for i in range(len(target_rows)):
        rid = rally_uids[i]
        N = int(next_sns[i])
        grp = rally_to_rows.get(rid)
        if grp is None or len(grp) == 0:
            out_rows.append({**_empty_features(),
                             **{k: 0 for k in feature_names()
                                if k.startswith("shot_lag")}})
            continue
        prefix = grp[grp["strikeNumber"] < N]
        if len(prefix) > 0 and int(prefix["strikeNumber"].max()) >= N:
            max_src_violations += 1
        out_rows.append(_compute_v2_row_features(prefix))

    assert max_src_violations == 0, \
        (f"features_server_v2: {max_src_violations} rows used a source "
         "strikeNumber >= next_strikeNumber.")

    fnames = feature_names()
    feat_df = pd.DataFrame(out_rows, columns=fnames)
    label = "train" if is_train else "test"
    print(f"  [features_server_v2] {label}: built {len(feat_df)} rows "
          f"x {len(fnames)} cols  empty_prefix={int(feat_df['empty_prefix'].sum())}")
    return feat_df


def build_test_per_rally_features_v2(test_df: pd.DataFrame) -> tuple:
    """Per-rally features for test (v1 + lag block)."""
    rally_uids = []
    out_rows = []
    cols = ["rally_uid", "strikeNumber", "gamePlayerId", "actionId", "pointId",
            "handId", "spinId", "strengthId", "scoreSelf", "scoreOther",
            "numberGame", "sex"]
    raw = test_df[cols].copy()
    raw["strikeNumber"] = raw["strikeNumber"].astype(int)
    raw["gamePlayerId"] = raw["gamePlayerId"].astype(int)
    for rid, grp in raw.groupby("rally_uid", sort=False):
        grp = grp.sort_values("strikeNumber").reset_index(drop=True)
        rally_uids.append(rid)
        out_rows.append(_compute_v2_row_features(grp))
    fnames = feature_names()
    feat_df = pd.DataFrame(out_rows, columns=fnames)
    print(f"  [features_server_v2] test per-rally: built {len(feat_df)} rallies "
          f"x {len(fnames)} cols")
    return feat_df, np.array(rally_uids)
