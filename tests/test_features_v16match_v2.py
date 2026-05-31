"""Unit tests for R-032 v2 (Codex APPROVE_WITH_FIXES, 8 fixes)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from features_v16match_v2 import (  # noqa: E402
    V16MATCH_V2_ADDED_COLUMNS,
    audit_no_banned_names_v16match_v2,
    _shannon_entropy_freq,
    _make_match_pair_key,
    _build_rally_to_match_pair,
    _deterministic_select,
    _aggregate_pair_features,
    PREFIX_CAP_K,
    MIN_OTHER_RALLIES,
)


def _build_match(match_id, rallies_spec, base_uid=1000):
    """rallies_spec: [(player_a, player_b, actions, points), ...]"""
    rows = []
    uid = base_uid
    for pa, pb, actions, points in rallies_spec:
        for sn, (a, p) in enumerate(zip(actions, points), start=1):
            rows.append({
                "rally_uid": uid, "match": match_id, "strikeNumber": sn,
                "actionId": a, "pointId": p, "handId": 1,
                "strengthId": 1, "spinId": 1, "positionId": 0,
                "scoreSelf": 0, "scoreOther": 0,
                "sex": 1, "numberGame": 1, "rally_id": uid,
                "serverGetPoint": 0, "gamePlayerId": pa,
                "gamePlayerOtherId": pb, "strikeId": 1 if sn == 1 else 4,
            })
        uid += 1
    return pd.DataFrame(rows)


# ---- Column inventory ---------------------------------------------------

def test_v2_columns_33():
    assert len(V16MATCH_V2_ADDED_COLUMNS) == 33

def test_v2_columns_unique():
    assert len(set(V16MATCH_V2_ADDED_COLUMNS)) == 33

def test_v2_no_family_c_in_model():
    """Codex P2-5: Family C dropped from model features."""
    family_c = ["match_pair_other_count", "match_pair_other_avg_rally_len"]
    for c in family_c:
        assert c not in V16MATCH_V2_ADDED_COLUMNS


# ---- Banned audit -------------------------------------------------------

def test_banned_audit_passes():
    audit_no_banned_names_v16match_v2(list(V16MATCH_V2_ADDED_COLUMNS))

def test_banned_audit_catches_sgp():
    with pytest.raises(ValueError, match="Banned"):
        audit_no_banned_names_v16match_v2(
            list(V16MATCH_V2_ADDED_COLUMNS) + ["match_other_avg_serverGetPoint"])

def test_banned_audit_catches_family_c():
    with pytest.raises(ValueError, match="Banned"):
        audit_no_banned_names_v16match_v2(
            list(V16MATCH_V2_ADDED_COLUMNS) + ["match_pair_other_count"])


# ---- Codex P1-1: match_pair grouping ------------------------------------

def test_match_pair_key_is_unordered():
    """(A, B) and (B, A) must produce the same key."""
    assert _make_match_pair_key(1, 10, 20) == _make_match_pair_key(1, 20, 10)


def test_match_pair_key_includes_match():
    """Different matches with same player pair = different keys."""
    assert _make_match_pair_key(1, 10, 20) != _make_match_pair_key(2, 10, 20)


def test_rally_to_match_pair_disambiguates_multi_player_match():
    """Codex P1-1: a single `match` with 3+ players splits into multiple pairs."""
    # Match 100 has rallies between players (10, 20), (10, 30), (20, 30)
    df = pd.concat([
        _build_match(100, [(10, 20, [1, 2, 3], [1, 2, 3])], base_uid=1000),
        _build_match(100, [(10, 30, [1, 2, 3], [1, 2, 3])], base_uid=1001),
        _build_match(100, [(20, 30, [1, 2, 3], [1, 2, 3])], base_uid=1002),
    ])
    pair_map = _build_rally_to_match_pair(df)
    assert pair_map[1000] != pair_map[1001]  # different pairs in same match
    assert pair_map[1001] != pair_map[1002]
    # All keys contain match id 100
    for v in pair_map.values():
        assert v.startswith("100|")


# ---- Codex P2-6: deterministic subsample --------------------------------

def test_deterministic_select_reproducible():
    """Two calls with same args return identical lists."""
    a = _deterministic_select([1, 2, 3, 4, 5, 6, 7, 8], target_uid=99, k=3)
    b = _deterministic_select([1, 2, 3, 4, 5, 6, 7, 8], target_uid=99, k=3)
    assert a == b


def test_deterministic_select_different_targets():
    """Different target_uid produces (likely) different subsamples."""
    a = _deterministic_select(list(range(20)), target_uid=10, k=5)
    b = _deterministic_select(list(range(20)), target_uid=11, k=5)
    # Should be different (probabilistically; 20-choose-5 = many options)
    assert a != b


def test_deterministic_select_k_ge_len():
    """If k >= len, return all."""
    out = _deterministic_select([1, 2, 3], target_uid=99, k=10)
    assert sorted(out) == [1, 2, 3]


# ---- Codex P1-4: prefix-length cap K=3 ----------------------------------

def test_prefix_cap_applied():
    """Only first K=3 prefix shots of each other rally contribute."""
    # Build a match with 2 rallies, each 10 shots long
    # Rally 0: shots 1-9 are action 5, shot 10 is action 17 (last, dropped)
    # Rally 1: shots 1-9 are action 8, shot 10 is action 17
    actions_long = [5] * 9 + [17]
    actions2_long = [8] * 9 + [17]
    points = [1] * 10
    df = pd.concat([
        _build_match(100, [(10, 20, actions_long, points)], base_uid=2000),
        _build_match(100, [(10, 20, actions2_long, points)], base_uid=2001),
    ])
    # min_other=0 to allow 2-rally pair test
    rally_to_pair = _build_rally_to_match_pair(df)
    aggs = _aggregate_pair_features(df, rally_to_pair, prefix_cap=3, min_other=0)
    # Rally 2000's aggregation should see only first 3 prefix shots of rally 2001
    # = action 8 × 3 shots (capped from 9), NOT 9.
    ac = aggs[2000]["action_counts"]
    assert ac[8] == 3, f"Expected 3 (capped), got {ac[8]}"
    # Action 5 from rally 2000 should NOT appear (LORO excludes own rally)
    assert ac[5] == 0


def test_loro_excludes_target():
    """Target rally's own data not in its features."""
    df = pd.concat([
        _build_match(100, [(10, 20, [1, 2, 3, 4], [1, 2, 3, 4])], base_uid=3000),
        _build_match(100, [(10, 20, [5, 6, 7, 8], [5, 6, 7, 8])], base_uid=3001),
        _build_match(100, [(10, 20, [9, 10, 11, 12], [1, 2, 3, 4])], base_uid=3002),
    ])
    rally_to_pair = _build_rally_to_match_pair(df)
    aggs = _aggregate_pair_features(df, rally_to_pair, prefix_cap=10, min_other=0)
    # Rally 3000's features should NOT include action 1, 2, 3 (its own)
    # (sn=4 is dropped per rally-last rule)
    ac3000 = aggs[3000]["action_counts"]
    assert ac3000[1] == 0
    assert ac3000[2] == 0
    assert ac3000[3] == 0
    # Should include actions 5, 6, 7 from rally 3001 (sn=8 dropped)
    assert ac3000[5] == 1
    assert ac3000[6] == 1
    assert ac3000[7] == 1


def test_max_other_cap_actually_applied():
    """Codex P1 (2026-05-22): when n_other > max_other, deterministic
    select to cap. The fast LORO-subtract path must NOT be used."""
    # Build 30 rallies in a single pair (1 match, 1 player pair)
    rallies = []
    for r in range(30):
        # Each rally has 4 shots; action varies per rally for distinct counts
        rallies.append((10, 20, [r % 19] * 4, [1, 2, 3, 4]))
    df = _build_match(100, rallies, base_uid=10000)
    rally_to_pair = _build_rally_to_match_pair(df)
    # Cap at 10
    aggs = _aggregate_pair_features(df, rally_to_pair, prefix_cap=3,
                                     min_other=0, max_other=10)
    # All rallies have n_other_total = 29, n_other_used = 10 (capped)
    for rid in [10000, 10010, 10029]:
        assert aggs[rid]["n_other_total"] == 29, f"rally {rid}: n_other_total={aggs[rid]['n_other_total']}"
        assert aggs[rid]["n_other_used"] == 10, f"rally {rid}: n_other_used={aggs[rid]['n_other_used']} (should be 10)"


def test_max_other_cap_deterministic():
    """Two runs with same args produce identical aggregates (deterministic
    selection, no RNG stream order)."""
    rallies = [(10, 20, [r % 19] * 4, [1, 2, 3, 4]) for r in range(25)]
    df = _build_match(100, rallies, base_uid=11000)
    pair_map = _build_rally_to_match_pair(df)
    a = _aggregate_pair_features(df, pair_map, prefix_cap=3, min_other=0, max_other=12)
    b = _aggregate_pair_features(df, pair_map, prefix_cap=3, min_other=0, max_other=12)
    # Same rally's action_counts must match exactly
    np.testing.assert_array_equal(a[11000]["action_counts"], b[11000]["action_counts"])
    np.testing.assert_array_equal(a[11010]["action_counts"], b[11010]["action_counts"])


def test_max_other_no_cap_below_threshold():
    """If n_other <= max_other, no capping; uses fast LORO subtract."""
    rallies = [(10, 20, [r] * 4, [1, 2, 3, 4]) for r in range(5)]
    df = _build_match(100, rallies, base_uid=12000)
    pair_map = _build_rally_to_match_pair(df)
    aggs = _aggregate_pair_features(df, pair_map, prefix_cap=3, min_other=0, max_other=22)
    # n_other = 4, max_other = 22 → no cap, n_other_used = 4
    for rid in [12000, 12001, 12002, 12003, 12004]:
        assert aggs[rid]["n_other_total"] == 4
        assert aggs[rid]["n_other_used"] == 4


def test_min_other_guard_zeros_features():
    """If n_other < MIN_OTHER_RALLIES, features must be zero."""
    # Match with only 1 rally — n_other = 0
    df = _build_match(100, [(10, 20, [1, 2, 3, 4], [1, 2, 3, 4])], base_uid=4000)
    rally_to_pair = _build_rally_to_match_pair(df)
    aggs = _aggregate_pair_features(df, rally_to_pair, prefix_cap=3,
                                     min_other=MIN_OTHER_RALLIES)
    # n_other_total = 0 (only 1 rally), below min
    assert aggs[4000]["n_other_total"] == 0
    assert aggs[4000]["n_other_used"] == 0
    assert aggs[4000]["action_counts"].sum() == 0


# ---- Shannon entropy ----------------------------------------------------

def test_entropy_zero_on_empty():
    assert _shannon_entropy_freq(np.zeros(19)) == 0.0


def test_entropy_uniform_2():
    assert _shannon_entropy_freq(np.array([5.0, 5.0])) == pytest.approx(np.log(2), abs=1e-9)
