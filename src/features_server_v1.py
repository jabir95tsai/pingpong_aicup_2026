"""features_server_v1 — prefix-only safe-feature module for serverGetPoint.

R-006 implementation per Codex APPROVE_WITH_FIXES (2026-05-09):

1. NO `gamePlayerId`, `gamePlayerOtherId`, or any player-ID target encoding.
2. Every feature derived from visible prefix rows only:
   `strikeNumber < next_strikeNumber`. Asserted at build time.
3. Server / receiver defined from visible prefix only: server = shooter at
   strike 1 (shot N=1 is always serve); receiver = the other player.
   NEVER inferred from the final winner / final hitter / rally-end aggregate.
4. Categorical histograms / proportions for `actionId`, `pointId`, `strengthId`,
   `spinId`, `handId`. NO "mean of categorical id".
5. Prefix counts (server / receiver / total) included with the diagnostic flag
   so we can run a separate counts-only AUC report.
6. Score features taken from strike 1 only (the rally-start visible row).

FORBIDDEN (NOT in this feature set):
- `n_shots` of full rally
- `strikeNumber` of last shot (terminal parity)
- Aggregates over rows with `strikeNumber >= N` (rally suffix)
- Mode / aggregate of `actionId` over the entire rally
- Anything that reveals who hit the final shot
- gamePlayerId / gamePlayerOtherId as features (allowed only to identify
  server/receiver internally, never as encoded model inputs)
"""
import numpy as np
import pandas as pd

# Bin counts (chosen to cover known label space).
N_ACTION_BIN = 15  # actionId 0..14 (15-class action eval space)
N_POINT_BIN = 10   # pointId 0..9
N_HAND_BIN = 3     # 0=none, 1=FH, 2=BH
N_SPIN_BIN = 6     # 0=none, 1..5 = spin types
N_STRENGTH_BIN = 4 # 0=none, 1=strong, 2=mid, 3=weak

CATEGORICAL_HISTS = [
    ("actionId", N_ACTION_BIN),
    ("pointId", N_POINT_BIN),
    ("handId", N_HAND_BIN),
    ("spinId", N_SPIN_BIN),
    ("strengthId", N_STRENGTH_BIN),
]


def _empty_features() -> dict:
    """Default feature dict for empty prefix rows."""
    feats = {}
    for col, nbin in CATEGORICAL_HISTS:
        for role in ("srv", "rcv"):
            for i in range(nbin):
                feats[f"{role}_{col}_p{i}"] = 0.0
    feats["server_prior_count"] = 0
    feats["receiver_prior_count"] = 0
    feats["total_prior_count"] = 0
    feats["score_self_start"] = 0
    feats["score_other_start"] = 0
    feats["score_diff_start"] = 0
    feats["numberGame"] = 0
    feats["sex"] = 0
    feats["empty_prefix"] = 1
    return feats


def _compute_row_features(prefix_df: pd.DataFrame) -> dict:
    """Compute per-row server-head features from a rally's prefix shots."""
    if len(prefix_df) == 0:
        return _empty_features()

    feats = {}

    # Server = shooter at strike 1; receiver = the other player.
    strike1 = prefix_df[prefix_df["strikeNumber"] == 1]
    if len(strike1) > 0:
        server_id = int(strike1["gamePlayerId"].iloc[0])
    else:
        # Strike 1 not in prefix — fall back to first available row's shooter.
        server_id = int(prefix_df.iloc[0]["gamePlayerId"])

    is_server_row = prefix_df["gamePlayerId"] == server_id
    server_rows = prefix_df[is_server_row]
    receiver_rows = prefix_df[~is_server_row]

    feats["server_prior_count"] = int(len(server_rows))
    feats["receiver_prior_count"] = int(len(receiver_rows))
    feats["total_prior_count"] = int(len(prefix_df))
    feats["empty_prefix"] = 0

    # Categorical histograms (proportions, sum to 1 if non-empty).
    for col, nbin in CATEGORICAL_HISTS:
        for role, role_rows in [("srv", server_rows), ("rcv", receiver_rows)]:
            hist = np.zeros(nbin, dtype=np.float32)
            if len(role_rows) > 0:
                vals = role_rows[col].astype(int).to_numpy()
                # Clip to valid range; out-of-range values are dropped.
                in_range = (vals >= 0) & (vals < nbin)
                if in_range.any():
                    counts = np.bincount(vals[in_range], minlength=nbin)
                    hist = counts.astype(np.float32) / max(counts.sum(), 1)
            for i in range(nbin):
                feats[f"{role}_{col}_p{i}"] = float(hist[i])

    # Score features from strike 1 (rally start) — Codex recommendation.
    if len(strike1) > 0:
        feats["score_self_start"] = int(strike1["scoreSelf"].iloc[0])
        feats["score_other_start"] = int(strike1["scoreOther"].iloc[0])
    else:
        feats["score_self_start"] = int(prefix_df.iloc[0]["scoreSelf"])
        feats["score_other_start"] = int(prefix_df.iloc[0]["scoreOther"])
    feats["score_diff_start"] = feats["score_self_start"] - feats["score_other_start"]

    # Rally-level metadata (constant within rally; from any prefix row).
    feats["numberGame"] = int(prefix_df["numberGame"].iloc[0])
    feats["sex"] = int(prefix_df["sex"].iloc[0])

    return feats


def feature_names() -> list:
    """Stable feature column order."""
    cols = []
    for col, nbin in CATEGORICAL_HISTS:
        for role in ("srv", "rcv"):
            for i in range(nbin):
                cols.append(f"{role}_{col}_p{i}")
    cols += [
        "server_prior_count", "receiver_prior_count", "total_prior_count",
        "score_self_start", "score_other_start", "score_diff_start",
        "numberGame", "sex", "empty_prefix",
    ]
    return cols


def count_only_features() -> list:
    """Subset used for the count-only AUC diagnostic (Codex requirement)."""
    return ["server_prior_count", "receiver_prior_count", "total_prior_count",
            "empty_prefix"]


def next_strike_only_features() -> list:
    """If we expose next_strikeNumber as a feature column, list it here for
    the diagnostic AUC. We DO NOT include it in the main model — Codex
    flagged it as a parity proxy."""
    return []


def build_features_server_v1(target_rows: pd.DataFrame,
                              raw_df: pd.DataFrame,
                              is_train: bool = True) -> pd.DataFrame:
    """Per-row prefix-only feature build.

    Args:
      target_rows: DataFrame with columns ['rally_uid', 'next_strikeNumber'].
        One row per output sample.
      raw_df: All raw shot rows (the cleaned full train_df or test_df).
      is_train: For logging only.

    Returns:
      DataFrame with one row per target_rows row, columns = feature_names().
      Asserts that no source row with strikeNumber >= next_strikeNumber leaked
      into any feature.
    """
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
            out_rows.append(_empty_features())
            continue
        prefix = grp[grp["strikeNumber"] < N]
        # Diagnostic invariant.
        if len(prefix) > 0 and int(prefix["strikeNumber"].max()) >= N:
            max_src_violations += 1
        out_rows.append(_compute_row_features(prefix))

    assert max_src_violations == 0, \
        (f"features_server_v1: {max_src_violations} rows used a source "
         "strikeNumber >= next_strikeNumber. Prefix-only invariant violated.")

    fnames = feature_names()
    feat_df = pd.DataFrame(out_rows, columns=fnames)
    label = "train" if is_train else "test"
    print(f"  [features_server_v1] {label}: built {len(feat_df)} rows "
          f"x {len(fnames)} cols  empty_prefix={int(feat_df['empty_prefix'].sum())}")
    return feat_df


def build_test_per_rally_features(test_df: pd.DataFrame) -> tuple:
    """For test: one feature row per rally, using ALL visible test rows as
    the prefix (i.e. the prediction is for the next-shot-after-the-last-visible
    of each test rally).
    """
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
        out_rows.append(_compute_row_features(grp))
    fnames = feature_names()
    feat_df = pd.DataFrame(out_rows, columns=fnames)
    print(f"  [features_server_v1] test per-rally: built {len(feat_df)} rallies "
          f"x {len(fnames)} cols")
    return feat_df, np.array(rally_uids)
