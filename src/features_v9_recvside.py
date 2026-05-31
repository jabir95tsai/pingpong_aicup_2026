"""features_v9_recvside: V9 + recv_side_est  (R-211 handedness/side-consistency).

Adds ONE integer feature ``recv_side_est in {0, 1, 2}`` to the v9 feature set.
It estimates the dominant point-SIDE tendency of the *receiver* of target shot
N, from that receiver's OWN prior shots in the SAME rally.

Motivation (R-211, 2026-05-31 gap reassessment):
  pointId's FH/BH axis is receiver-relative (handedness). The earlier
  ``recv_hand_est`` (features_v9_recvhand) estimated this from prior handId
  MODE — a weak proxy, since every player uses both FH and BH strokes; it gave
  only +0.0005 OV (though it did break the BH_short F1=0 floor). A probe showed
  the receiver's prior point-SIDE is a stronger axis signal: conditioning the
  next point-side on the striker's own prior-side majority shifts P(FH) by
  +0.062 / -0.085 (spread +0.147). recv_side_est exposes exactly that.

Hard-rule compliance (mirrors the Codex R-001 recvhand discipline):
  1. For a feature row with next_strikeNumber=N, the receiver is the shooter of
     strike N-1 (gamePlayerId at N-1). We only ever read rows strikeNumber < N.
  2. Source rows asserted to never include strikeNumber >= N (prefix-only).
  3. pointId values outside the FH/BH side sets are ignored (mid/miss carry no
     side). Mode taken over {FH-side, BH-side}; tie or no observation -> 0.
  4. Single integer feature only — no count/length companion (overfit guard).
  5. Within-rally + positional only: groups by gamePlayerId to isolate the
     receiver's own prior shots IN THIS RALLY. No cross-rally aggregation, no
     player-identity profile -> transfers to de-identified test.
  6. Prints train/test value distribution.
"""
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_v9 import (
    build_features_v9, compute_global_stats_v9, get_feature_names_v9,
)

# Re-export for symmetry with other feature modules.
compute_global_stats_v9_recvside = compute_global_stats_v9

FH_SIDE = (1, 4, 7)   # FH short/half/long
BH_SIDE = (3, 6, 9)   # BH short/half/long


def get_feature_names_v9_recvside(feat_df: pd.DataFrame) -> list:
    base = get_feature_names_v9(feat_df)
    if "recv_side_est" in feat_df.columns and "recv_side_est" not in base:
        return base + ["recv_side_est"]
    return base


def _compute_recv_side_est(feat_df: pd.DataFrame, raw_df: pd.DataFrame) -> np.ndarray:
    """Per-row recv_side_est in {0, 1, 2}.

    For each (rally_uid, next_strikeNumber=N) the receiver is the shooter of
    strike N-1. Over all rows strikeNumber < N in the same rally where the
    shooter matches that receiver and pointId is on a side, return the side
    mode (1=FH-side, 2=BH-side; 0 on tie / no observation).
    """
    raw = raw_df[["rally_uid", "strikeNumber", "gamePlayerId", "pointId"]].copy()
    raw["strikeNumber"] = raw["strikeNumber"].astype(int)
    raw["gamePlayerId"] = raw["gamePlayerId"].astype(int)
    raw["pointId"] = raw["pointId"].astype(int)

    fh_set = np.array(FH_SIDE)
    bh_set = np.array(BH_SIDE)

    cache: dict = {}
    max_src_violations = 0

    for rid, grp in raw.groupby("rally_uid", sort=False):
        grp = grp.sort_values("strikeNumber").reset_index(drop=True)
        sns = grp["strikeNumber"].to_numpy(dtype=int)
        gpids = grp["gamePlayerId"].to_numpy(dtype=int)
        pts = grp["pointId"].to_numpy(dtype=int)

        for k in range(len(grp)):
            N = int(sns[k]) + 1
            receiver_id = int(gpids[k])

            base_mask = (sns < N) & (gpids == receiver_id)
            if base_mask.sum() == 0:
                cache[(rid, N)] = 0
                continue

            max_src_sn = int(sns[base_mask].max())
            if max_src_sn >= N:
                max_src_violations += 1

            p_vals = pts[base_mask]
            c_fh = int(np.isin(p_vals, fh_set).sum())
            c_bh = int(np.isin(p_vals, bh_set).sum())
            if c_fh > c_bh:
                cache[(rid, N)] = 1
            elif c_bh > c_fh:
                cache[(rid, N)] = 2
            else:
                cache[(rid, N)] = 0  # tie / no side observed -> unknown

    assert max_src_violations == 0, (
        f"recv_side_est: {max_src_violations} rows sourced from strikeNumber >= "
        "next_strikeNumber. Prefix-only invariant violated.")

    n = len(feat_df)
    out = np.zeros(n, dtype=np.int8)
    rally_arr = feat_df["rally_uid"].to_numpy()
    nsn_arr = feat_df["next_strikeNumber"].to_numpy(dtype=int)
    for i in range(n):
        out[i] = cache.get((rally_arr[i], int(nsn_arr[i])), 0)
    return out


def build_features_v9_recvside(df: pd.DataFrame, is_train: bool,
                               global_stats_v9: dict,
                               raw_df: pd.DataFrame = None) -> pd.DataFrame:
    """V9 features + recv_side_est."""
    feat_df = build_features_v9(df, is_train=is_train,
                                 global_stats_v9=global_stats_v9,
                                 raw_df=raw_df)
    if raw_df is None:
        raw_df = df

    recv_side = _compute_recv_side_est(feat_df, raw_df)
    feat_df["recv_side_est"] = recv_side.astype(np.int8)

    n = len(recv_side)
    pct0 = (recv_side == 0).mean() * 100.0
    pct1 = (recv_side == 1).mean() * 100.0
    pct2 = (recv_side == 2).mean() * 100.0
    label = "train" if is_train else "test"
    print(f"  [recv_side_est] {label} n={n}  unknown(0)={pct0:.1f}%  "
          f"FH-side(1)={pct1:.1f}%  BH-side(2)={pct2:.1f}%")

    return feat_df
