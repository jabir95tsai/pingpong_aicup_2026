"""features_v9_recvhand: V9 + recv_hand_est.

Adds one integer feature `recv_hand_est ∈ {0, 1, 2}` to the v9 feature set.
This estimates the dominant hand of the **receiver** of the target shot N, so
the model can resolve receiver-relative pointId labels (FH/BH grid axis).

Per Codex R-001 review (2026-05-08):

1. For a feature row with ``next_strikeNumber = N`` in rally R, the target
   receiver is ``gamePlayerId`` at strike ``N-1`` in R. We never read raw row
   N — only rows ``strikeNumber < N``.
2. Source rows are filtered to ``strikeNumber < next_strikeNumber`` and
   asserted to never include rows ``>= N``.
3. ``handId`` values 0 are ignored. The mode is taken over ``{1, 2}`` only;
   on a tie or no valid prior hand, emit 0.
4. No count / length companion features are added — only the single
   ``recv_hand_est`` integer.
5. ``train_v14.py`` must explicitly opt into this feature set via
   ``--feature-set v9_recvhand``; nothing is implicit.
6. The build step prints the train/test value distribution (incl.
   percent-unknown).
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
compute_global_stats_v9_recvhand = compute_global_stats_v9


def get_feature_names_v9_recvhand(feat_df: pd.DataFrame) -> list:
    base = get_feature_names_v9(feat_df)
    if "recv_hand_est" in feat_df.columns and "recv_hand_est" not in base:
        return base + ["recv_hand_est"]
    return base


def _compute_recv_hand_est(feat_df: pd.DataFrame, raw_df: pd.DataFrame) -> np.ndarray:
    """Per-row recv_hand_est in {0, 1, 2}.

    For each (rally_uid, next_strikeNumber=N) the target receiver is the
    shooter of strike N-1 (i.e. ``gamePlayerId`` at strike N-1). We then
    look at all rows ``strikeNumber < N`` in the same rally where the shooter
    matches that receiver and ``handId in {1, 2}``, and return the mode
    (or 0 on tie / no observation).
    """
    raw = raw_df[["rally_uid", "strikeNumber", "gamePlayerId", "handId"]].copy()
    raw["strikeNumber"] = raw["strikeNumber"].astype(int)
    raw["gamePlayerId"] = raw["gamePlayerId"].astype(int)
    raw["handId"] = raw["handId"].astype(int)

    # Per-rally cache keyed by target N.
    cache: dict = {}
    max_src_violations = 0

    for rid, grp in raw.groupby("rally_uid", sort=False):
        grp = grp.sort_values("strikeNumber").reset_index(drop=True)
        sns = grp["strikeNumber"].to_numpy(dtype=int)
        gpids = grp["gamePlayerId"].to_numpy(dtype=int)
        hands = grp["handId"].to_numpy(dtype=int)

        for k in range(len(grp)):
            # Target shot N comes after the row at sns[k]. We only consider
            # N = sns[k] + 1 — the natural next-strike successor.
            N = int(sns[k]) + 1
            receiver_id = int(gpids[k])

            # Source rows: strikeNumber < N AND shooter == receiver_id AND
            # handId in {1, 2}.
            src_mask = (sns < N) & (gpids == receiver_id) & np.isin(hands, [1, 2])
            if src_mask.sum() == 0:
                cache[(rid, N)] = 0
                continue

            # Diagnostic: max source strikeNumber must be < N.
            max_src_sn = int(sns[src_mask].max())
            if max_src_sn >= N:
                max_src_violations += 1

            h_vals = hands[src_mask]
            c1 = int((h_vals == 1).sum())
            c2 = int((h_vals == 2).sum())
            if c1 > c2:
                cache[(rid, N)] = 1
            elif c2 > c1:
                cache[(rid, N)] = 2
            else:
                cache[(rid, N)] = 0  # tie → unknown

    assert max_src_violations == 0, (
        f"recv_hand_est: {max_src_violations} rows sourced from strikeNumber >= "
        "next_strikeNumber. Prefix-only invariant violated.")

    n = len(feat_df)
    out = np.zeros(n, dtype=np.int8)
    rally_arr = feat_df["rally_uid"].to_numpy()
    nsn_arr = feat_df["next_strikeNumber"].to_numpy(dtype=int)
    for i in range(n):
        out[i] = cache.get((rally_arr[i], int(nsn_arr[i])), 0)

    return out


def build_features_v9_recvhand(df: pd.DataFrame, is_train: bool,
                               global_stats_v9: dict,
                               raw_df: pd.DataFrame = None) -> pd.DataFrame:
    """V9 features + recv_hand_est."""
    feat_df = build_features_v9(df, is_train=is_train,
                                 global_stats_v9=global_stats_v9,
                                 raw_df=raw_df)
    if raw_df is None:
        raw_df = df

    recv_hand = _compute_recv_hand_est(feat_df, raw_df)
    feat_df["recv_hand_est"] = recv_hand.astype(np.int8)

    # Distribution log (Codex requirement #6).
    n = len(recv_hand)
    pct0 = (recv_hand == 0).mean() * 100.0
    pct1 = (recv_hand == 1).mean() * 100.0
    pct2 = (recv_hand == 2).mean() * 100.0
    label = "train" if is_train else "test"
    print(f"  [recv_hand_est] {label} n={n}  unknown(0)={pct0:.1f}%  "
          f"right(1)={pct1:.1f}%  left(2)={pct2:.1f}%")

    return feat_df
