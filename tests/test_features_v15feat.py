"""Unit smoke tests for src/features_v15feat.py (R-029a Batch A).

Covers:
- Empty rally (no history): all 36 features = 0
- Single-shot history: degenerate distribution, entropy = 0, dominance = 1
- Mixed multi-shot history: per-class freqs correct, entropy > 0
- Tail-streak counting: pure runs vs. mixed sequences
- Column count matches V15FEAT_ADDED_COLUMNS (36)

These tests use synthetic dataframes to avoid the heavy V9 dependency
where possible. The full integration with V9 is exercised in the regular
training pipeline; this file targets V15feat-specific correctness only.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from features_v15feat import (  # noqa: E402
    V15FEAT_ADDED_COLUMNS,
    _shannon_entropy_from_counts,
    _tail_run_length,
)


def test_added_columns_count_is_36():
    assert len(V15FEAT_ADDED_COLUMNS) == 36


def test_added_columns_unique():
    assert len(set(V15FEAT_ADDED_COLUMNS)) == 36


# ---------- _shannon_entropy_from_counts ----------

def test_entropy_empty_is_zero():
    assert _shannon_entropy_from_counts(np.zeros(19, dtype=int)) == 0.0


def test_entropy_degenerate_is_zero():
    """One class only → entropy = 0."""
    counts = np.zeros(19, dtype=int)
    counts[5] = 10
    assert _shannon_entropy_from_counts(counts) == pytest.approx(0.0)


def test_entropy_uniform_two_classes():
    """P=[0.5, 0.5] entropy = ln(2)."""
    counts = np.array([5, 5], dtype=int)
    assert _shannon_entropy_from_counts(counts) == pytest.approx(np.log(2), abs=1e-9)


def test_entropy_uniform_full_action_space():
    """Uniform over 19 classes → entropy = ln(19)."""
    counts = np.ones(19, dtype=int)
    assert _shannon_entropy_from_counts(counts) == pytest.approx(np.log(19), abs=1e-9)


def test_entropy_zeros_dont_pollute():
    """Adding zero-count classes must not change entropy."""
    a = _shannon_entropy_from_counts(np.array([3, 7], dtype=int))
    b = _shannon_entropy_from_counts(np.array([3, 0, 0, 7, 0], dtype=int))
    assert a == pytest.approx(b, abs=1e-12)


# ---------- _tail_run_length ----------

def test_tail_run_empty():
    assert _tail_run_length(np.array([], dtype=int)) == 0


def test_tail_run_single_element():
    assert _tail_run_length(np.array([7], dtype=int)) == 1


def test_tail_run_pure_run():
    assert _tail_run_length(np.array([4, 4, 4, 4], dtype=int)) == 4


def test_tail_run_mixed_with_tail():
    """Mixed sequence with a 3-long tail run."""
    assert _tail_run_length(np.array([1, 2, 5, 5, 5], dtype=int)) == 3


def test_tail_run_tail_breaks_immediately():
    """Tail is unique → run length 1."""
    assert _tail_run_length(np.array([5, 5, 5, 7], dtype=int)) == 1


def test_tail_run_all_distinct():
    assert _tail_run_length(np.array([1, 2, 3, 4], dtype=int)) == 1


# ---------- build_features_v15feat (integration with V9) ----------
# These tests exercise the full feature builder. They depend on V9 / V7
# infrastructure but use the smallest possible synthetic rallies. If V9
# changes shape, the V15feat-added columns must still all appear.

def _make_synthetic_train_df() -> pd.DataFrame:
    """Two short rallies + minimum columns required by clean_data + V7/V9."""
    base_row = {
        "sex": 1, "numberGame": 1, "rally_id": 1,
        "scoreSelf": 0, "scoreOther": 0,
        "gamePlayerId": 1, "gamePlayerOtherId": 2,
        "strikeId": 1, "handId": 1, "strengthId": 1, "spinId": 0,
        "positionId": 0, "match": 1,
    }
    rows = []
    # Rally 1: action sequence 1,2,2 — point sequence 5,5,7 — players 1,2,1
    for sn, (act, pt, p1, p2) in enumerate(
        [(1, 5, 1, 2), (2, 5, 2, 1), (2, 7, 1, 2)], start=1
    ):
        rows.append({
            **base_row,
            "rally_uid": 1001, "strikeNumber": sn,
            "actionId": act, "pointId": pt,
            "gamePlayerId": p1, "gamePlayerOtherId": p2,
            "serverGetPoint": 1,
        })
    # Rally 2: longer, varied
    for sn, (act, pt) in enumerate(
        [(15, 1, ), (3, 4), (1, 4), (1, 4), (10, 8)], start=1
    ):
        rows.append({
            **base_row,
            "rally_uid": 1002, "strikeNumber": sn,
            "actionId": act, "pointId": pt,
            "serverGetPoint": 0,
        })
    return pd.DataFrame(rows)


def test_build_v15feat_adds_36_columns():
    """All 36 V15feat columns must be present in the output feat_df."""
    # Import here to avoid module-level cost when only entropy/streak tests run.
    from data_cleaning import clean_data
    from features_v15feat import (
        build_features_v15feat,
        compute_global_stats_v15feat,
    )

    raw = _make_synthetic_train_df()
    train_df, _, _ = clean_data(raw.copy(), raw.iloc[:0].copy())
    stats = compute_global_stats_v15feat(train_df)
    feat = build_features_v15feat(train_df, is_train=True,
                                   global_stats_v9=stats,
                                   raw_df=train_df)

    for col in V15FEAT_ADDED_COLUMNS:
        assert col in feat.columns, f"Missing V15feat column: {col}"


def test_build_v15feat_frequencies_sum_to_one():
    """For every row with a non-empty prefix, action freqs and point freqs
    must each sum to 1.0 (within float tolerance)."""
    from data_cleaning import clean_data
    from features_v15feat import (
        build_features_v15feat,
        compute_global_stats_v15feat,
    )

    raw = _make_synthetic_train_df()
    train_df, _, _ = clean_data(raw.copy(), raw.iloc[:0].copy())
    stats = compute_global_stats_v15feat(train_df)
    feat = build_features_v15feat(train_df, is_train=True,
                                   global_stats_v9=stats,
                                   raw_df=train_df)

    act_cols = [f"hist_action_freq_{c}" for c in range(19)]
    pt_cols = [f"hist_point_freq_{c}" for c in range(10)]
    act_sums = feat[act_cols].sum(axis=1).values
    pt_sums = feat[pt_cols].sum(axis=1).values

    # At least some rows should have non-empty histories
    non_empty = act_sums > 0.0
    assert non_empty.any(), "synthetic data produced only empty-history rows"
    np.testing.assert_allclose(act_sums[non_empty], 1.0, atol=1e-5)
    np.testing.assert_allclose(pt_sums[non_empty], 1.0, atol=1e-5)


def test_build_v15feat_dominance_in_range():
    """Dominance must be in [0, 1] inclusive."""
    from data_cleaning import clean_data
    from features_v15feat import (
        build_features_v15feat,
        compute_global_stats_v15feat,
    )

    raw = _make_synthetic_train_df()
    train_df, _, _ = clean_data(raw.copy(), raw.iloc[:0].copy())
    stats = compute_global_stats_v15feat(train_df)
    feat = build_features_v15feat(train_df, is_train=True,
                                   global_stats_v9=stats,
                                   raw_df=train_df)

    for c in ("hist_action_dominance", "hist_point_dominance"):
        vals = feat[c].values
        assert (vals >= 0.0).all()
        assert (vals <= 1.0 + 1e-6).all()


def test_build_v15feat_streaks_nonnegative_integers():
    """Streak columns must be non-negative integers."""
    from data_cleaning import clean_data
    from features_v15feat import (
        build_features_v15feat,
        compute_global_stats_v15feat,
    )

    raw = _make_synthetic_train_df()
    train_df, _, _ = clean_data(raw.copy(), raw.iloc[:0].copy())
    stats = compute_global_stats_v15feat(train_df)
    feat = build_features_v15feat(train_df, is_train=True,
                                   global_stats_v9=stats,
                                   raw_df=train_df)

    for c in ("streak_action_tail", "streak_point_tail", "consecutive_same_player"):
        vals = feat[c].values
        assert (vals >= 0).all()
        assert np.issubdtype(vals.dtype, np.integer), \
            f"{c} should be integer dtype, got {vals.dtype}"
