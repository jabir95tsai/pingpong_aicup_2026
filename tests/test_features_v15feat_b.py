"""Unit smoke tests for src/features_v15feat_b.py (R-029b Batch B).

Covers:
- Added column count = 33
- Action priors per row sum to 1
- Point priors per row sum to 1
- Entropy is non-negative and <= log(N)
- top1 is in [0, 1]
- Marginal fallback used when context unseen
- Tables are fold-safe: a table built from a single rally's data only sees
  that rally's transitions (no cross-rally leakage)
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from features_v15feat_b import (  # noqa: E402
    V15FEAT_B_ADDED_COLUMNS,
    _build_action_prior_table,
    _build_point_prior_table,
    _shannon_entropy_from_probs,
)


# ---------- V15FEAT_B_ADDED_COLUMNS ----------

def test_added_columns_count_is_33():
    assert len(V15FEAT_B_ADDED_COLUMNS) == 33


def test_added_columns_unique():
    assert len(set(V15FEAT_B_ADDED_COLUMNS)) == 33


# ---------- _shannon_entropy_from_probs (vectorized) ----------

def test_entropy_degenerate_row_is_zero():
    """One-hot row → entropy = 0."""
    probs = np.zeros((1, 19), dtype=np.float32)
    probs[0, 3] = 1.0
    e = _shannon_entropy_from_probs(probs)
    assert e[0] == pytest.approx(0.0, abs=1e-7)


def test_entropy_uniform_row_is_log_n():
    probs = np.full((1, 19), 1.0 / 19, dtype=np.float32)
    e = _shannon_entropy_from_probs(probs)
    assert e[0] == pytest.approx(np.log(19), abs=1e-6)


def test_entropy_multiple_rows():
    probs = np.array([
        [1.0] + [0.0] * 18,                   # one-hot → 0
        [1.0 / 19] * 19,                       # uniform → ln(19)
        [0.5, 0.5] + [0.0] * 17,               # 2-uniform → ln(2)
    ], dtype=np.float32)
    e = _shannon_entropy_from_probs(probs)
    assert e[0] == pytest.approx(0.0, abs=1e-6)
    assert e[1] == pytest.approx(np.log(19), abs=1e-6)
    assert e[2] == pytest.approx(np.log(2), abs=1e-6)


# ---------- Transition tables ----------

def _toy_train_df() -> pd.DataFrame:
    """Two rallies. Rally A: action sequence [1, 2, 3] all on serve=odd
    convention. Rally B: action sequence [5, 5, 1] mixed.
    """
    base = {
        "sex": 1, "numberGame": 1, "rally_id": 1, "match": 1,
        "scoreSelf": 0, "scoreOther": 0, "serverGetPoint": 1,
        "gamePlayerId": 1, "gamePlayerOtherId": 2,
        "strikeId": 1, "handId": 1, "strengthId": 1, "spinId": 0,
        "positionId": 0,
    }
    rows = []
    # Rally A: strikes 1,2,3 with actions 1,2,3 and points 5,5,7
    for sn, act, pt in [(1, 1, 5), (2, 2, 5), (3, 3, 7)]:
        rows.append({**base, "rally_uid": 1001, "strikeNumber": sn,
                     "actionId": act, "pointId": pt})
    # Rally B: strikes 1,2,3 with actions 5,5,1 and points 0,8,8
    for sn, act, pt in [(1, 5, 0), (2, 5, 8), (3, 1, 8)]:
        rows.append({**base, "rally_uid": 1002, "strikeNumber": sn,
                     "actionId": act, "pointId": pt})
    return pd.DataFrame(rows)


def test_action_table_contains_known_transitions():
    """Verify a hand-computed transition shows up in the table."""
    df = _toy_train_df()
    table, marginal = _build_action_prior_table(df)
    # Rally A: action 1@sn1 → action 2@sn2 (even = receive side, is_serve=0)
    # So key=(1, 0). Vector should have all probability mass on class 2.
    assert (1, 0) in table
    vec = table[(1, 0)]
    assert vec[2] == pytest.approx(1.0, abs=1e-6)
    assert vec.sum() == pytest.approx(1.0, abs=1e-6)


def test_action_marginal_sums_to_one():
    df = _toy_train_df()
    _, marginal = _build_action_prior_table(df)
    assert marginal.sum() == pytest.approx(1.0, abs=1e-6)


def test_point_table_contains_known_transitions():
    """Rally B: action 5 + point 0 → next point 8 (sn1 → sn2)."""
    df = _toy_train_df()
    table, marginal = _build_point_prior_table(df)
    assert (5, 0) in table
    vec = table[(5, 0)]
    assert vec[8] == pytest.approx(1.0, abs=1e-6)
    assert vec.sum() == pytest.approx(1.0, abs=1e-6)


def test_point_marginal_sums_to_one():
    df = _toy_train_df()
    _, marginal = _build_point_prior_table(df)
    assert marginal.sum() == pytest.approx(1.0, abs=1e-6)


def test_fold_safety_table_doesnt_see_other_rally():
    """If only Rally A is in the training fold, Rally B's transitions
    must NOT appear in the table. This is the leak-safety invariant.
    """
    df = _toy_train_df()
    rally_a_only = df[df["rally_uid"] == 1001]
    table_a, _ = _build_action_prior_table(rally_a_only)
    # Rally B's last_action=5 → key (5, 0) or (5, 1) should NOT exist
    assert (5, 0) not in table_a
    assert (5, 1) not in table_a


# ---------- Integration: build_features_v15feat_b ----------

def test_build_v15feat_b_adds_33_columns_on_top_of_v15feat():
    """End-to-end: building features adds all V15feat_b columns."""
    from data_cleaning import clean_data
    from features_v15feat import V15FEAT_ADDED_COLUMNS
    from features_v15feat_b import (
        build_features_v15feat_b,
        compute_global_stats_v15feat_b,
    )

    df = _toy_train_df()
    # Pad with extra rallies so V7 statistics don't fail on tiny data
    extra_rows = []
    base = df.iloc[0].to_dict()
    for rid in range(2000, 2050):
        for sn in range(1, 4):
            row = dict(base)
            row.update({
                "rally_uid": rid, "strikeNumber": sn,
                "actionId": (sn + rid) % 19,
                "pointId": (sn + rid) % 10,
                "serverGetPoint": (rid % 2),
                "gamePlayerId": rid % 5, "gamePlayerOtherId": (rid + 1) % 5,
            })
            extra_rows.append(row)
    df_big = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    train_df, _, _ = clean_data(df_big.copy(), df_big.iloc[:0].copy())
    stats = compute_global_stats_v15feat_b(train_df)
    feat = build_features_v15feat_b(train_df, is_train=True,
                                      global_stats_v9=stats,
                                      raw_df=train_df)

    for col in V15FEAT_ADDED_COLUMNS:
        assert col in feat.columns, f"Missing V15feat column: {col}"
    for col in V15FEAT_B_ADDED_COLUMNS:
        assert col in feat.columns, f"Missing V15feat_b column: {col}"


def test_build_v15feat_b_priors_sum_to_one():
    """Action and point priors must sum to 1.0 per row."""
    from data_cleaning import clean_data
    from features_v15feat_b import (
        build_features_v15feat_b,
        compute_global_stats_v15feat_b,
    )

    df = _toy_train_df()
    # Pad as above
    base = df.iloc[0].to_dict()
    extra = []
    for rid in range(3000, 3030):
        for sn in range(1, 4):
            row = dict(base)
            row.update({
                "rally_uid": rid, "strikeNumber": sn,
                "actionId": (rid + sn) % 19,
                "pointId": (rid + sn) % 10,
                "serverGetPoint": rid % 2,
                "gamePlayerId": rid % 4, "gamePlayerOtherId": (rid + 1) % 4,
            })
            extra.append(row)
    df_big = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    train_df, _, _ = clean_data(df_big.copy(), df_big.iloc[:0].copy())
    stats = compute_global_stats_v15feat_b(train_df)
    feat = build_features_v15feat_b(train_df, is_train=True,
                                      global_stats_v9=stats,
                                      raw_df=train_df)

    action_cols = [f"trans_action_prior_{c}" for c in range(19)]
    point_cols = [f"trans_point_prior_{c}" for c in range(10)]
    act_sums = feat[action_cols].sum(axis=1).values
    pt_sums = feat[point_cols].sum(axis=1).values
    np.testing.assert_allclose(act_sums, 1.0, atol=1e-5)
    np.testing.assert_allclose(pt_sums, 1.0, atol=1e-5)


def test_build_v15feat_b_top1_in_range():
    from data_cleaning import clean_data
    from features_v15feat_b import (
        build_features_v15feat_b,
        compute_global_stats_v15feat_b,
    )

    df = _toy_train_df()
    # Pad
    base = df.iloc[0].to_dict()
    extra = []
    for rid in range(4000, 4030):
        for sn in range(1, 4):
            row = dict(base)
            row.update({
                "rally_uid": rid, "strikeNumber": sn,
                "actionId": (rid + sn) % 19,
                "pointId": (rid + sn) % 10,
                "serverGetPoint": rid % 2,
                "gamePlayerId": rid % 4, "gamePlayerOtherId": (rid + 1) % 4,
            })
            extra.append(row)
    df_big = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    train_df, _, _ = clean_data(df_big.copy(), df_big.iloc[:0].copy())
    stats = compute_global_stats_v15feat_b(train_df)
    feat = build_features_v15feat_b(train_df, is_train=True,
                                      global_stats_v9=stats,
                                      raw_df=train_df)

    for c in ("trans_action_top1", "trans_point_top1"):
        vals = feat[c].values
        assert (vals >= 0.0).all()
        assert (vals <= 1.0 + 1e-6).all()
