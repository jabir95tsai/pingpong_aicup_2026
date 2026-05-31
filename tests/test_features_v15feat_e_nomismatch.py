"""Tests for v15feat_e_nomismatch (R-070 ablation, Codex 2026-05-24).

All v15feat_e invariants except the mismatch-proxy ones (since dropped).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from features_v15feat_e_nomismatch import (  # noqa: E402
    V15FEAT_E_NOMISMATCH_ADDED_COLUMNS,
    compute_global_stats_v15feat_e_nomismatch,
    build_features_v15feat_e_nomismatch,
    _point_to_side,
    _point_to_depth,
)


def _make_rally(rally_uid, match_id, shots, sgp=1):
    rows = []
    for sn, shot in enumerate(shots, start=1):
        rows.append({
            "rally_uid": rally_uid, "match": match_id, "strikeNumber": sn,
            "strikeId": shot.get("strikeId", 4),
            "actionId": shot["actionId"], "pointId": shot["pointId"],
            "handId": shot.get("handId", 1), "strengthId": shot.get("strengthId", 2),
            "spinId": shot.get("spinId", 1), "positionId": shot.get("positionId", 1),
            "gamePlayerId": shot.get("gamePlayerId", 10),
            "gamePlayerOtherId": shot.get("gamePlayerOtherId", 20),
            "numberGame": 1, "rally_id": rally_uid % 100, "sex": 1,
            "scoreSelf": 0, "scoreOther": 0, "serverGetPoint": sgp,
        })
    return rows


def _make_dataset(specs):
    rows = []
    for rally_uid, match_id, shots in specs:
        rows.extend(_make_rally(rally_uid, match_id, shots))
    return pd.DataFrame(rows)


# Test 1: exact count = 5
def test_exact_feature_count():
    assert len(V15FEAT_E_NOMISMATCH_ADDED_COLUMNS) == 5
    expected = {"last_point_side", "last_point_depth", "last_point_valid",
                "last_position_valid", "last_outgoing_lateral_gap"}
    assert set(V15FEAT_E_NOMISMATCH_ADDED_COLUMNS) == expected
    # Mismatch features must NOT be present
    forbidden = {"stroke_position_mismatch_proxy", "mismatch_AND_far_gap"}
    for f in forbidden:
        assert f not in V15FEAT_E_NOMISMATCH_ADDED_COLUMNS, (
            f"Dropped feature '{f}' must not exist in nomismatch ablation"
        )


# Test 2: 2D pointId decomp
def test_point_decomposition():
    assert _point_to_side(1) == 1 and _point_to_depth(1) == 1
    assert _point_to_side(5) == 2 and _point_to_depth(5) == 2
    assert _point_to_side(9) == 3 and _point_to_depth(9) == 3
    assert _point_to_side(0) == 0 and _point_to_depth(0) == 0


# Test 3: missingness flags honest
def test_missingness_flags_honest():
    df = _make_dataset([
        (100, 1, [
            {"actionId": 15, "pointId": 0, "handId": 1, "positionId": 0},
            {"actionId": 10, "pointId": 5, "handId": 1, "positionId": 2},
            {"actionId": 1,  "pointId": 7, "handId": 2, "positionId": 3},
        ]),
    ])
    stats = compute_global_stats_v15feat_e_nomismatch(df)
    feat = build_features_v15feat_e_nomismatch(df, is_train=True,
                                                  global_stats_v9=stats, raw_df=df)
    rows_2 = feat[feat["next_strikeNumber"] == 2]
    assert len(rows_2) >= 1
    for _, r in rows_2.iterrows():
        assert r["last_point_valid"] == 0.0
        assert r["last_position_valid"] == 0.0
        assert r["last_outgoing_lateral_gap"] == 0.0
        assert r["last_point_side"] == 0.0
        assert r["last_point_depth"] == 0.0


# Test 4: no NaN/Inf
def test_no_nan_inf():
    df = _make_dataset([
        (100, 1, [
            {"actionId": 15, "pointId": 0, "handId": 0, "positionId": 0},
            {"actionId": 1,  "pointId": 5, "handId": 1, "positionId": 2},
            {"actionId": 10, "pointId": 7, "handId": 2, "positionId": 3},
        ]),
    ])
    stats = compute_global_stats_v15feat_e_nomismatch(df)
    feat = build_features_v15feat_e_nomismatch(df, is_train=True,
                                                 global_stats_v9=stats, raw_df=df)
    for col in V15FEAT_E_NOMISMATCH_ADDED_COLUMNS:
        vals = feat[col].to_numpy(dtype=np.float64)
        assert np.isfinite(vals).all()


# Test 5: no SGP read
def test_no_sgp_read():
    df = _make_dataset([
        (100, 1, [
            {"actionId": 15, "pointId": 5, "handId": 1, "positionId": 1},
            {"actionId": 10, "pointId": 3, "handId": 2, "positionId": 3},
            {"actionId": 1,  "pointId": 7, "handId": 1, "positionId": 2},
        ]),
    ])
    stats_a = compute_global_stats_v15feat_e_nomismatch(df)
    feat_a = build_features_v15feat_e_nomismatch(df, is_train=True,
                                                   global_stats_v9=stats_a, raw_df=df)
    df_flipped = df.copy()
    df_flipped["serverGetPoint"] = 1 - df_flipped["serverGetPoint"].astype(int)
    stats_b = compute_global_stats_v15feat_e_nomismatch(df_flipped)
    feat_b = build_features_v15feat_e_nomismatch(df_flipped, is_train=True,
                                                    global_stats_v9=stats_b, raw_df=df_flipped)
    for col in V15FEAT_E_NOMISMATCH_ADDED_COLUMNS:
        np.testing.assert_array_equal(feat_a[col].to_numpy(), feat_b[col].to_numpy(),
                                       err_msg=f"{col} changed when SGP flipped")


# Test 6: prefix-only
def test_prefix_only():
    base = [
        {"strikeId": 1, "actionId": 15, "pointId": 5, "handId": 1, "positionId": 1},
        {"strikeId": 2, "actionId": 10, "pointId": 3, "handId": 2, "positionId": 3},
        {"strikeId": 4, "actionId": 1, "pointId": 7, "handId": 1, "positionId": 2},
    ]
    mutated = [
        {"strikeId": 1, "actionId": 15, "pointId": 5, "handId": 1, "positionId": 1},
        {"strikeId": 2, "actionId": 10, "pointId": 3, "handId": 2, "positionId": 3},
        {"strikeId": 4, "actionId": 13, "pointId": 1, "handId": 2, "positionId": 3},
    ]
    df_a = _make_dataset([(100, 1, base), (200, 1, base)])
    df_b = _make_dataset([(100, 1, mutated), (200, 1, mutated)])
    stats = compute_global_stats_v15feat_e_nomismatch(df_a)
    feat_a = build_features_v15feat_e_nomismatch(df_a, is_train=True, global_stats_v9=stats, raw_df=df_a)
    feat_b = build_features_v15feat_e_nomismatch(df_b, is_train=True, global_stats_v9=stats, raw_df=df_b)
    rows_a = feat_a[feat_a["next_strikeNumber"] == 3]
    rows_b = feat_b[feat_b["next_strikeNumber"] == 3]
    for col in V15FEAT_E_NOMISMATCH_ADDED_COLUMNS:
        np.testing.assert_array_equal(rows_a[col].to_numpy(), rows_b[col].to_numpy())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
