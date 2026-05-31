"""R-070 invariant tests for v15feat_e (Codex APPROVE_WITH_FIXES 2026-05-24).

Codex required tests:
1. Prefix-only construction (no target-row read)
2. Sparsity invariants: pointId=0 / positionId=0 produce explicit missing flags,
   NOT fake distances
3. Exact feature count (7)
4. No NaN/Inf
5. No serverGetPoint column ever read
6. No alternating-player movement (i.e. no raw position_change feature)
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from features_v15feat_e import (  # noqa: E402
    V15FEAT_E_ADDED_COLUMNS,
    MISMATCH_PAIRS,
    compute_global_stats_v15feat_e,
    build_features_v15feat_e,
    _point_to_side,
    _point_to_depth,
)


def _make_rally(rally_uid, match_id, shots, sgp=1):
    rows = []
    for sn, shot in enumerate(shots, start=1):
        rows.append({
            "rally_uid": rally_uid,
            "match": match_id,
            "strikeNumber": sn,
            "strikeId": shot.get("strikeId", 4),
            "actionId": shot["actionId"],
            "pointId": shot["pointId"],
            "handId": shot.get("handId", 1),
            "strengthId": shot.get("strengthId", 2),
            "spinId": shot.get("spinId", 1),
            "positionId": shot.get("positionId", 1),
            "gamePlayerId": shot.get("gamePlayerId", 10),
            "gamePlayerOtherId": shot.get("gamePlayerOtherId", 20),
            "numberGame": 1,
            "rally_id": rally_uid % 100,
            "sex": 1,
            "scoreSelf": 0,
            "scoreOther": 0,
            "serverGetPoint": sgp,
        })
    return rows


def _make_dataset(specs):
    rows = []
    for rally_uid, match_id, shots in specs:
        rows.extend(_make_rally(rally_uid, match_id, shots))
    return pd.DataFrame(rows)


# ─── Test 1: Exact feature count (Codex fix #4) ──────────────────────────────

def test_exact_feature_count():
    assert len(V15FEAT_E_ADDED_COLUMNS) == 7
    expected = {
        "stroke_position_mismatch_proxy", "last_point_side", "last_point_depth",
        "last_point_valid", "last_position_valid", "last_outgoing_lateral_gap",
        "mismatch_AND_far_gap",
    }
    assert set(V15FEAT_E_ADDED_COLUMNS) == expected


# ─── Test 2: 2D pointId decomposition correctness ────────────────────────────

def test_point_to_side_depth():
    # pointId 1=FH-short → side=1, depth=1
    assert _point_to_side(1) == 1
    assert _point_to_depth(1) == 1
    # pointId 5=mid-half → side=2, depth=2
    assert _point_to_side(5) == 2
    assert _point_to_depth(5) == 2
    # pointId 9=BH-long → side=3, depth=3
    assert _point_to_side(9) == 3
    assert _point_to_depth(9) == 3
    # pointId 0=missing → both 0
    assert _point_to_side(0) == 0
    assert _point_to_depth(0) == 0


# ─── Test 3: Missingness flags honest (Codex fix #5) ─────────────────────────

def test_missingness_flags_honest():
    """pointId=0 / positionId=0 must produce explicit missing flags, NOT fake distances."""
    df = _make_dataset([
        # Rally with positionId=0 and pointId=0 in last shot of prefix
        (100, 1, [
            {"actionId": 15, "pointId": 0, "handId": 1, "positionId": 0},  # missing both
            {"actionId": 10, "pointId": 5, "handId": 1, "positionId": 2},  # valid
            {"actionId": 1,  "pointId": 7, "handId": 2, "positionId": 3},  # target
        ]),
    ])
    stats = compute_global_stats_v15feat_e(df)
    feat = build_features_v15feat_e(df, is_train=True, global_stats_v9=stats, raw_df=df)
    # For target row (predicting shot 3), prefix = [shot1, shot2]. Last in prefix = shot 2.
    # shot 2: positionId=2, pointId=5 → both valid
    # Find rows with next_strikeNumber == 3
    rows_3 = feat[feat["next_strikeNumber"] == 3]
    assert len(rows_3) >= 1
    for _, r in rows_3.iterrows():
        assert r["last_point_valid"] == 1.0, "shot 2 has pointId=5 valid"
        assert r["last_position_valid"] == 1.0, "shot 2 has positionId=2 valid"

    # For target row predicting shot 2, prefix = [shot 1]. shot 1: positionId=0, pointId=0
    rows_2 = feat[feat["next_strikeNumber"] == 2]
    assert len(rows_2) >= 1
    for _, r in rows_2.iterrows():
        assert r["last_point_valid"] == 0.0, "shot 1 has pointId=0 missing"
        assert r["last_position_valid"] == 0.0, "shot 1 has positionId=0 missing"
        assert r["last_outgoing_lateral_gap"] == 0.0, "gap must be 0 when missing, NOT fake"
        assert r["last_point_side"] == 0.0
        assert r["last_point_depth"] == 0.0


# ─── Test 4: No NaN / Inf (Codex fix #4) ────────────────────────────────────

def test_no_nan_inf():
    df = _make_dataset([
        (100, 1, [
            {"actionId": 15, "pointId": 0, "handId": 0, "positionId": 0},
            {"actionId": 1,  "pointId": 5, "handId": 1, "positionId": 2},
            {"actionId": 10, "pointId": 7, "handId": 2, "positionId": 3},
        ]),
        (200, 1, [
            {"actionId": 16, "pointId": 4, "handId": 1, "positionId": 1},  # mismatch
            {"actionId": 5,  "pointId": 7, "handId": 1, "positionId": 2},
        ]),
    ])
    stats = compute_global_stats_v15feat_e(df)
    feat = build_features_v15feat_e(df, is_train=True, global_stats_v9=stats, raw_df=df)
    for col in V15FEAT_E_ADDED_COLUMNS:
        assert col in feat.columns
        vals = feat[col].to_numpy(dtype=np.float64)
        assert np.isfinite(vals).all(), f"Column {col} has non-finite values"


# ─── Test 5: No SGP read (Codex fix #4) ─────────────────────────────────────

def test_no_sgp_read():
    """Mutating serverGetPoint must not change feature outputs."""
    df = _make_dataset([
        (100, 1, [
            {"actionId": 15, "pointId": 5, "handId": 1, "positionId": 1},  # mismatch
            {"actionId": 10, "pointId": 3, "handId": 2, "positionId": 3},  # mismatch
            {"actionId": 1,  "pointId": 7, "handId": 1, "positionId": 2},
        ]),
    ])
    stats_a = compute_global_stats_v15feat_e(df)
    feat_a = build_features_v15feat_e(df, is_train=True, global_stats_v9=stats_a, raw_df=df)

    df_flipped = df.copy()
    df_flipped["serverGetPoint"] = 1 - df_flipped["serverGetPoint"].astype(int)
    stats_b = compute_global_stats_v15feat_e(df_flipped)
    feat_b = build_features_v15feat_e(df_flipped, is_train=True,
                                       global_stats_v9=stats_b, raw_df=df_flipped)

    for col in V15FEAT_E_ADDED_COLUMNS:
        np.testing.assert_array_equal(
            feat_a[col].to_numpy(), feat_b[col].to_numpy(),
            err_msg=f"Column {col} changed when SGP flipped — SGP leak detected"
        )


# ─── Test 6: Prefix-only construction (Codex fix #4) ────────────────────────

def test_prefix_only_construction():
    """Mutating the TARGET shot must not change v15feat_e features."""
    base_rally = [
        {"strikeId": 1, "actionId": 15, "pointId": 5, "handId": 1, "positionId": 1},  # mismatch
        {"strikeId": 2, "actionId": 10, "pointId": 3, "handId": 2, "positionId": 3},  # mismatch
        # TARGET (sn=3) — mutating this must not affect features
        {"strikeId": 4, "actionId": 1, "pointId": 7, "handId": 1, "positionId": 2},
    ]
    mutated_rally = [
        {"strikeId": 1, "actionId": 15, "pointId": 5, "handId": 1, "positionId": 1},
        {"strikeId": 2, "actionId": 10, "pointId": 3, "handId": 2, "positionId": 3},
        # Mutated TARGET — different action, point, hand, position
        {"strikeId": 4, "actionId": 13, "pointId": 1, "handId": 2, "positionId": 3},
    ]
    df_a = _make_dataset([(100, 1, base_rally), (200, 1, base_rally)])
    df_b = _make_dataset([(100, 1, mutated_rally), (200, 1, mutated_rally)])

    stats = compute_global_stats_v15feat_e(df_a)
    feat_a = build_features_v15feat_e(df_a, is_train=True, global_stats_v9=stats, raw_df=df_a)
    feat_b = build_features_v15feat_e(df_b, is_train=True, global_stats_v9=stats, raw_df=df_b)

    # Rows predicting shot 3 (target) must have IDENTICAL features for both inputs
    rows_a = feat_a[feat_a["next_strikeNumber"] == 3]
    rows_b = feat_b[feat_b["next_strikeNumber"] == 3]
    assert len(rows_a) > 0
    for col in V15FEAT_E_ADDED_COLUMNS:
        np.testing.assert_array_equal(
            rows_a[col].to_numpy(), rows_b[col].to_numpy(),
            err_msg=f"Column {col} changed when target was mutated — prefix-only invariant broken"
        )


# ─── Test 7: Mismatch proxy correctness (Codex fix #1) ──────────────────────

def test_mismatch_proxy_correctness():
    """stroke_position_mismatch_proxy = 1 iff (handId, positionId) in {(1,1), (2,3)}."""
    df = _make_dataset([
        # rally 1: last prefix shot (handId=1, positionId=1) → mismatch
        (100, 1, [
            {"actionId": 15, "pointId": 5, "handId": 1, "positionId": 1},
            {"actionId": 1,  "pointId": 7, "handId": 1, "positionId": 2},
        ]),
        # rally 2: last prefix shot (handId=2, positionId=3) → mismatch
        (200, 1, [
            {"actionId": 16, "pointId": 4, "handId": 2, "positionId": 3},
            {"actionId": 10, "pointId": 3, "handId": 1, "positionId": 1},
        ]),
        # rally 3: last prefix shot (handId=1, positionId=2) → NOT mismatch
        (300, 1, [
            {"actionId": 15, "pointId": 5, "handId": 1, "positionId": 2},
            {"actionId": 5,  "pointId": 7, "handId": 1, "positionId": 1},
        ]),
    ])
    stats = compute_global_stats_v15feat_e(df)
    feat = build_features_v15feat_e(df, is_train=True, global_stats_v9=stats, raw_df=df)
    # For each rally, find target = shot 2 (predicting from prefix shot 1)
    for rally_uid, expected in [(100, 1.0), (200, 1.0), (300, 0.0)]:
        rows = feat[(feat["rally_uid"] == rally_uid) & (feat["next_strikeNumber"] == 2)]
        assert len(rows) >= 1, f"rally {rally_uid} missing target row"
        for _, r in rows.iterrows():
            assert r["stroke_position_mismatch_proxy"] == expected, (
                f"rally {rally_uid}: expected mismatch={expected}, got {r['stroke_position_mismatch_proxy']}"
            )


# ─── Test 8: No raw alternating-player movement feature (Codex fix #2) ──────

def test_no_alternating_player_movement_feature():
    """Verify position_change_in_prefix is NOT in the added columns (was originally
    proposed but dropped per Codex fix #2 because consecutive shots alternate hitters)."""
    forbidden = {"position_change_in_prefix", "position_change_mean", "raw_movement"}
    for f in forbidden:
        assert f not in V15FEAT_E_ADDED_COLUMNS, (
            f"Forbidden alternating-player movement feature '{f}' must not exist"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
