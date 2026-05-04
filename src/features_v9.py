"""Feature engineering V9: extends V7 with joint serve-receive priors.

V7 baseline: 1145 features.
V9 adds 25 features for SN=2 receive shots:

  Group 6 — joint (serve_act, serve_pt, sex) → receive priors:
    P(recv_act | serve_act, serve_pt, sex) → v9_recv_act_p0..p14  (15 probs)
    P(recv_pt  | serve_act, serve_pt, sex) → v9_recv_pt_p0..p9   (10 probs)

  For non-SN=2 rows: falls back to marginal P(·|serve_act, sex).
  V7 already has P(recv_act|serve_act, sex) as summary stats; V9 provides the
  full vector AND additionally conditions on serve_pt landing position.

All tables are fold-safe (computed from training fold only).
"""
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_v7 import (
    build_features_v7, compute_global_stats_v7, get_feature_names_v7,
)

N_ACT_TRAIN = 15   # receive action classes 0-14
N_PT = 10


def _build_v9_tables(train_df: pd.DataFrame) -> dict:
    """Build joint (serve_act, serve_pt, sex) → P(recv_act), P(recv_pt) tables."""
    joint_act    = {}
    joint_pt     = {}
    marginal_act = {}
    marginal_pt  = {}

    rallies = train_df.groupby("rally_uid", sort=False)
    for _, grp in rallies:
        grp = grp.sort_values("strikeNumber").reset_index(drop=True)
        sns  = grp["strikeNumber"].values.astype(int)
        acts = grp["actionId"].values.astype(int)
        pts  = grp["pointId"].values.astype(int)
        sex  = int(grp["sex"].iloc[0]) if "sex" in grp.columns else 1

        for i in range(1, len(grp)):
            if sns[i] != 2:
                continue
            serve_act = int(acts[i-1])
            serve_pt  = int(pts[i-1]) if 0 <= pts[i-1] < N_PT else 0
            recv_act  = min(int(acts[i]), N_ACT_TRAIN - 1)
            recv_pt   = int(pts[i]) if 0 <= pts[i] < N_PT else 0

            jkey = (serve_act, serve_pt, sex)
            mkey = (serve_act, sex)

            joint_act.setdefault(jkey, np.zeros(N_ACT_TRAIN))[recv_act] += 1
            joint_pt.setdefault(jkey,  np.zeros(N_PT))[recv_pt]          += 1
            marginal_act.setdefault(mkey, np.zeros(N_ACT_TRAIN))[recv_act] += 1
            marginal_pt.setdefault(mkey,  np.zeros(N_PT))[recv_pt]          += 1

    def _norm(d, size):
        uniform = np.full(size, 1.0 / size, dtype=np.float32)
        out = {}
        for k, v in d.items():
            s = v.sum()
            out[k] = (v / s).astype(np.float32) if s > 0 else uniform.copy()
        return out

    return {
        "joint_act":    _norm(joint_act,    N_ACT_TRAIN),
        "joint_pt":     _norm(joint_pt,     N_PT),
        "marginal_act": _norm(marginal_act, N_ACT_TRAIN),
        "marginal_pt":  _norm(marginal_pt,  N_PT),
    }


def compute_global_stats_v9(train_df: pd.DataFrame) -> dict:
    stats = compute_global_stats_v7(train_df)
    stats["v9_tables"] = _build_v9_tables(train_df)
    return stats


def get_feature_names_v9(feat_df: pd.DataFrame) -> list:
    return get_feature_names_v7(feat_df)


def build_features_v9(df: pd.DataFrame, is_train: bool,
                       global_stats_v9: dict,
                       raw_df: pd.DataFrame = None) -> pd.DataFrame:
    """Build V7 features + 25 joint serve-receive priors (fold-safe)."""
    feat_df = build_features_v7(df, is_train=is_train,
                                 global_stats_v7=global_stats_v9,
                                 raw_df=raw_df)
    if raw_df is None:
        raw_df = df

    tables = global_stats_v9["v9_tables"]
    n = len(feat_df)
    uniform_act = np.full(N_ACT_TRAIN, 1.0 / N_ACT_TRAIN, dtype=np.float32)
    uniform_pt  = np.full(N_PT, 1.0 / N_PT, dtype=np.float32)

    # next_strikeNumber in feat_df tells us which shot is being predicted.
    # SN=2 rows: serve context is the previous shot (lag-1).
    nsn     = feat_df["next_strikeNumber"].values.astype(int)
    sex_arr = feat_df["sex"].values.astype(int) if "sex" in feat_df.columns else np.ones(n, int)

    # Look up serve action/point via raw_df merge (same pattern as V7)
    shot_lookup = raw_df[["rally_uid", "strikeNumber", "actionId", "pointId"]].copy()
    shot_lookup["strikeNumber"] = shot_lookup["strikeNumber"].astype(int)

    merge_left = pd.DataFrame({
        "rally_uid":    feat_df["rally_uid"].values,
        "strikeNumber": nsn - 1,
    })
    merged = merge_left.merge(shot_lookup, on=["rally_uid", "strikeNumber"], how="left")
    serve_act_arr = merged["actionId"].fillna(-1).astype(int).values
    serve_pt_arr  = merged["pointId"].fillna(0).astype(int).values

    extra_act = np.zeros((n, N_ACT_TRAIN), dtype=np.float32)
    extra_pt  = np.zeros((n, N_PT),        dtype=np.float32)

    for i in range(n):
        if nsn[i] != 2:
            extra_act[i] = uniform_act
            extra_pt[i]  = uniform_pt
            continue

        serve_act = int(serve_act_arr[i])
        serve_pt  = int(serve_pt_arr[i]) if 0 <= serve_pt_arr[i] < N_PT else 0
        sex       = int(sex_arr[i])
        jkey = (serve_act, serve_pt, sex)
        mkey = (serve_act, sex)

        extra_act[i] = tables["joint_act"].get(
            jkey, tables["marginal_act"].get(mkey, uniform_act))
        extra_pt[i]  = tables["joint_pt"].get(
            jkey, tables["marginal_pt"].get(mkey, uniform_pt))

    for c in range(N_ACT_TRAIN):
        feat_df[f"v9_recv_act_p{c}"] = extra_act[:, c]
    for c in range(N_PT):
        feat_df[f"v9_recv_pt_p{c}"] = extra_pt[:, c]

    return feat_df
