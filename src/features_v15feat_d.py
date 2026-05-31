"""Feature engineering v15feat_d_core: v15feat + spin-aware features (R-064).

Builds on top of v15feat (R-029a Batch A prefix aggregates) with 13 spin-aware
features added per row. Domain motivation: in table tennis, the spin of the
incoming ball physically constrains the receiver's set of legal counter-shots
(heavy backspin → must lift; heavy topspin → must block/counter-loop). The
existing models see `lag_spinId` as an input but may not learn these
constraints from sparse data alone. Explicit features encode them.

R-064 Codex `APPROVE_WITH_FIXES` (2026-05-23) — fixes applied:
- Smaller "core" feature set (13, not the originally proposed 15): drop the
  two hard semantic flags `next_cannot_attack_due_to_backspin` and
  `next_must_block_due_to_topspin` per Codex fix #2.
- Group A spin priors use Dirichlet smoothing (alpha=20) against the global
  spin distribution per Codex fix #3.
- Group A stats are fold-train-only — computed via
  `compute_global_stats_v15feat_d` and passed through to the feature builder
  (the existing v9-family kwarg `global_stats_v9` carries them).
- All features are PREFIX-ONLY: derived from shots with
  `strikeNumber < next_strikeNumber`. The target-shot row is never read.
- serverGetPoint is never accessed.

Feature additions (13 total)

  Group A — Spin transition priors (5 features)
    P(next_spinId = c | last_actionId, last_positionId) for c in {1, 2, 3, 4, 5}
    Computed on fold-train only. Sparse bins smoothed via
    Dirichlet:  (counts + alpha * global_p) / (n_bin + alpha), alpha=20.
    Featurised at the LAST prefix shot's (actionId, positionId).

      prior_next_spin_class_1   — P(spin == 1: 上旋)
      prior_next_spin_class_2   — P(spin == 2: 下旋)
      prior_next_spin_class_3   — P(spin == 3: 不旋)
      prior_next_spin_class_4   — P(spin == 4: 側上旋)
      prior_next_spin_class_5   — P(spin == 5: 側下旋)

  Group B — Spin physics binary flags on the LAST PREFIX SHOT (4 features)
      last_was_heavy_backspin   — (last_actionId in {10, 11, 12}) AND (last_spinId == 2)
      last_was_heavy_topspin    — (last_actionId in {1, 2, 3}) AND (last_spinId == 1)
      last_was_sidespin         — last_spinId in {4, 5}
      last_was_no_spin          — last_spinId == 3

  Group C — Serve spin class one-hot (4 features, prefix-only)
    Derived from the SERVE shot (strikeNumber == 1) of the rally, always
    present in the prefix when next_strikeNumber >= 2. Captures the rally's
    serve-spin signature regardless of how deep the prediction sits.

      serve_topspin             — actionId in {15, 16, 18} AND spinId == 1
      serve_backspin            — (actionId == 15 AND spinId == 2) OR (actionId == 17)
      serve_sidespin            — spinId in {4, 5}
      serve_no_spin             — spinId == 3

Empty-prefix default: all 13 features = 0.0. Same for "no serve in prefix" on
Group C (only happens when next_strikeNumber == 1, i.e. we're predicting the
serve itself — degenerate case for spin features).

Test parity: see `tests/test_features_v15feat_d.py` for the invariants Codex
required (priors sum to 1, no NaN/Inf, exact feature count, no SGP read,
fold-train-only stats, prefix-only construction).
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

# ─── Constants ──────────────────────────────────────────────────────────────

# Spin class IDs per CLAUDE.md spinId: 0=無, 1=上旋, 2=下旋, 3=不旋, 4=側上旋, 5=側下旋
# We model only spin classes 1..5 (5 features); spin==0 is "unknown/无" sentinel,
# excluded from the prior distribution.
SPIN_CLASSES = (1, 2, 3, 4, 5)
N_SPIN = len(SPIN_CLASSES)

# Group B physics-flag action sets
HEAVY_BACKSPIN_ACTIONS = frozenset({10, 11, 12})  # 搓球, 擺短, 削球
HEAVY_TOPSPIN_ACTIONS = frozenset({1, 2, 3})       # 拉球, 反拉, 殺球

# Group C serve action IDs (per CLAUDE.md: 15=傳統, 16=勾手, 17=逆旋轉, 18=下蹲式)
SERVE_ACTIONS = frozenset({15, 16, 17, 18})

# Dirichlet smoothing strength (Codex fix #3)
DIRICHLET_ALPHA = 20.0

# Group A bin keys: (last_actionId, last_positionId) → P(next_spinId)
# actionId range: 0..18 (19 classes), positionId range: 0..3 (4 classes)
# → up to 76 bins. Many will be sparse, hence smoothing.
N_ACT_FULL = 19
N_POS = 4


# ─── Global stats (called per fold on fold-train rows only) ──────────────────

def compute_global_stats_v15feat_d(train_df: pd.DataFrame) -> dict:
    """Extend v15feat global stats with spin priors (Group A).

    Returns a dict with all v15feat keys plus:
      - 'spin_global_p':           np.ndarray shape (N_SPIN,) — overall P(spinId == c) on this train slice
      - 'spin_prior_smoothed':     dict keyed by (last_actionId, last_positionId)
                                   → np.ndarray shape (N_SPIN,) of smoothed priors
      - 'spin_prior_min_bin_n':    int — min raw count across non-empty bins
      - 'spin_prior_median_bin_n': int — median count across non-empty bins
      - 'spin_prior_unseen_rate':  float — fraction of (action, pos) bins with 0 obs

    Fold-safe: only reads `train_df` rows. Caller (`train_v14.py` fold loop)
    must pass tr_raw, not the full train_df.
    """
    stats = compute_global_stats_v15feat(train_df)

    # Build (last_action, last_position) → next_spinId pairs from train rallies.
    # "Last" = strikeNumber t; "next" = strikeNumber t+1 of the same rally.
    df = train_df.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
    df["_next_spinId"] = df.groupby("rally_uid", sort=False)["spinId"].shift(-1)
    # Drop rows where there is no next shot (last shot of the rally).
    df_pairs = df.dropna(subset=["_next_spinId"]).copy()
    df_pairs["_next_spinId"] = df_pairs["_next_spinId"].astype(int)
    # Restrict to legal spin classes 1..5 (drop 0 = unknown)
    df_pairs = df_pairs[df_pairs["_next_spinId"].isin(SPIN_CLASSES)]

    # Global prior P(spinId == c) over the train slice
    spin_global_counts = np.zeros(N_SPIN, dtype=np.float64)
    for idx, c in enumerate(SPIN_CLASSES):
        spin_global_counts[idx] = int((df_pairs["_next_spinId"] == c).sum())
    total = float(spin_global_counts.sum())
    if total > 0:
        spin_global_p = spin_global_counts / total
    else:
        spin_global_p = np.full(N_SPIN, 1.0 / N_SPIN, dtype=np.float64)

    # Per-bin smoothed prior
    spin_prior_smoothed: Dict[tuple, np.ndarray] = {}
    bin_ns: list = []
    grouped = df_pairs.groupby(["actionId", "positionId"])["_next_spinId"]
    for (a, p), vals in grouped:
        a_i, p_i = int(a), int(p)
        if a_i < 0 or a_i >= N_ACT_FULL or p_i < 0 or p_i >= N_POS:
            continue  # skip out-of-range bins
        counts = np.zeros(N_SPIN, dtype=np.float64)
        v_arr = vals.values.astype(np.int64)
        for idx, c in enumerate(SPIN_CLASSES):
            counts[idx] = float(np.sum(v_arr == c))
        n_bin = float(counts.sum())
        # Dirichlet smoothing: (counts + alpha * global_p) / (n + alpha)
        smoothed = (counts + DIRICHLET_ALPHA * spin_global_p) / (n_bin + DIRICHLET_ALPHA)
        spin_prior_smoothed[(a_i, p_i)] = smoothed.astype(np.float32)
        bin_ns.append(int(n_bin))

    # Coverage stats for the audit log
    total_bins = N_ACT_FULL * N_POS
    spin_prior_unseen_rate = 1.0 - (len(spin_prior_smoothed) / total_bins)
    spin_prior_min_bin_n = int(min(bin_ns)) if bin_ns else 0
    spin_prior_median_bin_n = int(np.median(bin_ns)) if bin_ns else 0

    stats["spin_global_p"] = spin_global_p.astype(np.float32)
    stats["spin_prior_smoothed"] = spin_prior_smoothed
    stats["spin_prior_min_bin_n"] = spin_prior_min_bin_n
    stats["spin_prior_median_bin_n"] = spin_prior_median_bin_n
    stats["spin_prior_unseen_rate"] = float(spin_prior_unseen_rate)
    # Codex fix #3 audit log
    print(
        f"  [v15feat_d] spin_prior coverage: "
        f"min_bin_n={spin_prior_min_bin_n}  median_bin_n={spin_prior_median_bin_n}  "
        f"unseen_rate={spin_prior_unseen_rate:.3f}  "
        f"observed_bins={len(spin_prior_smoothed)}/{N_ACT_FULL * N_POS}  "
        f"alpha={DIRICHLET_ALPHA}"
    )
    return stats


def get_feature_names_v15feat_d(feat_df: pd.DataFrame) -> list:
    """Return numeric feature column names. v15feat_d adds 13 to v15feat."""
    return get_feature_names_v15feat(feat_df)


def build_features_v15feat_d(
    df: pd.DataFrame,
    is_train: bool,
    global_stats_v9: dict,
    raw_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build v15feat features + 13 v15feat_d spin-aware features.

    Args:
        df: shot-level dataframe (post-`clean_data`).
        is_train: whether this is training data (label availability).
        global_stats_v9: dict returned by `compute_global_stats_v15feat_d`
            (named for v9-family kwarg convention).
        raw_df: raw shot-level dataframe (defaults to `df`). Used to look up
            prefix history per prediction target.

    Returns:
        feat_df with all v15feat columns + 13 new spin-aware columns. SGP is
        NEVER read; all derived from actionId / pointId / spinId / positionId
        / strikeNumber / strikeId on prefix shots.
    """
    feat_df = build_features_v15feat(
        df, is_train=is_train,
        global_stats_v9=global_stats_v9,
        raw_df=raw_df,
    )

    if raw_df is None:
        raw_df = df

    # Spin prior lookup table (fold-train-only stats)
    spin_global_p = np.asarray(
        global_stats_v9.get("spin_global_p", np.full(N_SPIN, 1.0 / N_SPIN, dtype=np.float32)),
        dtype=np.float32,
    )
    spin_prior_smoothed: Dict[tuple, np.ndarray] = global_stats_v9.get(
        "spin_prior_smoothed", {}
    )

    # Cache per-rally arrays sorted by strikeNumber for prefix lookups.
    rally_cache: dict[int, dict] = {}
    raw_sorted = raw_df.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
    for rid, grp in raw_sorted.groupby("rally_uid", sort=False):
        rally_cache[int(rid)] = {
            "strike":   grp["strikeNumber"].values.astype(np.int32),
            "act":      grp["actionId"].values.astype(np.int32),
            "pos":      grp["positionId"].values.astype(np.int32),
            "spin":     grp["spinId"].values.astype(np.int32),
            "strikeid": grp["strikeId"].values.astype(np.int32),
        }

    n_rows = len(feat_df)
    rid_arr = feat_df["rally_uid"].astype(np.int64).values
    nsn_arr = feat_df["next_strikeNumber"].astype(np.int32).values

    # Output buffers — exactly 13 features
    out_prior = np.zeros((n_rows, N_SPIN), dtype=np.float32)               # Group A: 5
    out_last_heavy_back = np.zeros(n_rows, dtype=np.float32)                # Group B: 4
    out_last_heavy_top = np.zeros(n_rows, dtype=np.float32)
    out_last_sidespin = np.zeros(n_rows, dtype=np.float32)
    out_last_nospin = np.zeros(n_rows, dtype=np.float32)
    out_serve_top = np.zeros(n_rows, dtype=np.float32)                     # Group C: 4
    out_serve_back = np.zeros(n_rows, dtype=np.float32)
    out_serve_side = np.zeros(n_rows, dtype=np.float32)
    out_serve_no = np.zeros(n_rows, dtype=np.float32)

    for i in range(n_rows):
        rid = int(rid_arr[i])
        next_sn = int(nsn_arr[i])
        cache = rally_cache.get(rid)
        if cache is None or next_sn <= 1:
            # next_sn=1 means we're predicting the serve itself — no history.
            continue
        # Prefix mask: strikeNumber < next_strikeNumber (strict — prefix-only).
        prefix_mask = cache["strike"] < next_sn
        if not prefix_mask.any():
            continue

        prefix_strike = cache["strike"][prefix_mask]
        prefix_act = cache["act"][prefix_mask]
        prefix_pos = cache["pos"][prefix_mask]
        prefix_spin = cache["spin"][prefix_mask]
        prefix_strikeid = cache["strikeid"][prefix_mask]

        # LAST prefix shot (the one immediately before the target).
        # The cache is sorted ascending, so last = max strikeNumber.
        last_idx = int(np.argmax(prefix_strike))
        last_action = int(prefix_act[last_idx])
        last_position = int(prefix_pos[last_idx])
        last_spin = int(prefix_spin[last_idx])

        # ── Group A: spin priors at (last_action, last_position) ─────────
        key = (last_action, last_position)
        if key in spin_prior_smoothed:
            out_prior[i] = spin_prior_smoothed[key]
        else:
            # Unseen bin → fall back to global prior (Dirichlet-smoothed already)
            out_prior[i] = spin_global_p

        # ── Group B: physics flags on last prefix shot ───────────────────
        if last_action in HEAVY_BACKSPIN_ACTIONS and last_spin == 2:
            out_last_heavy_back[i] = 1.0
        if last_action in HEAVY_TOPSPIN_ACTIONS and last_spin == 1:
            out_last_heavy_top[i] = 1.0
        if last_spin in (4, 5):
            out_last_sidespin[i] = 1.0
        if last_spin == 3:
            out_last_nospin[i] = 1.0

        # ── Group C: serve_spin_class — look at the SERVE shot (strikeNumber=1)
        # Only valid when prefix contains strikeNumber==1 (always true if
        # next_sn >= 2 and the rally starts at strikeNumber=1).
        serve_mask = prefix_strike == 1
        if not serve_mask.any():
            continue
        serve_idx = int(np.argmax(serve_mask))  # first match (should be only one)
        serve_action = int(prefix_act[serve_idx])
        serve_spin = int(prefix_spin[serve_idx])
        serve_strikeid = int(prefix_strikeid[serve_idx])
        # Defensive: only treat as serve if strikeId==1 (standard) AND actionId is a serve.
        if serve_strikeid != 1 or serve_action not in SERVE_ACTIONS:
            continue
        # Codex-flagged: prefix-only — we read serve_action / serve_spin from a row
        # whose strikeNumber == 1 < next_sn, satisfying the prefix-only constraint.
        if serve_action in {15, 16, 18} and serve_spin == 1:
            out_serve_top[i] = 1.0
        if (serve_action == 15 and serve_spin == 2) or (serve_action == 17):
            out_serve_back[i] = 1.0
        if serve_spin in (4, 5):
            out_serve_side[i] = 1.0
        if serve_spin == 3:
            out_serve_no[i] = 1.0

    # Materialize columns in stable order
    for k, c in enumerate(SPIN_CLASSES):
        feat_df[f"prior_next_spin_class_{c}"] = out_prior[:, k]
    feat_df["last_was_heavy_backspin"] = out_last_heavy_back
    feat_df["last_was_heavy_topspin"] = out_last_heavy_top
    feat_df["last_was_sidespin"] = out_last_sidespin
    feat_df["last_was_no_spin"] = out_last_nospin
    feat_df["serve_topspin"] = out_serve_top
    feat_df["serve_backspin"] = out_serve_back
    feat_df["serve_sidespin"] = out_serve_side
    feat_df["serve_no_spin"] = out_serve_no

    return feat_df


# Convenience: exact list of column names v15feat_d adds (for tests + audits)
V15FEAT_D_ADDED_COLUMNS = (
    [f"prior_next_spin_class_{c}" for c in SPIN_CLASSES]
    + [
        "last_was_heavy_backspin",
        "last_was_heavy_topspin",
        "last_was_sidespin",
        "last_was_no_spin",
        "serve_topspin",
        "serve_backspin",
        "serve_sidespin",
        "serve_no_spin",
    ]
)

assert len(V15FEAT_D_ADDED_COLUMNS) == 13, (
    f"v15feat_d column count drift: {len(V15FEAT_D_ADDED_COLUMNS)} != 13"
)
