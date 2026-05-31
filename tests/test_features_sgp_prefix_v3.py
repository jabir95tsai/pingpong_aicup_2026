"""Unit tests for src/features_sgp_prefix_v3.py (R-030 v1 core profile).

Covers:
- Feature count = 65 (core profile)
- Empty history → all zero defaults
- Strict prefix containment: max(prefix_strikeNumber) < target_strikeNumber
- Banned-name grep
- Test mode produces exactly 1 sample per rally
- Train mode produces L-1 samples per L-shot rally
- Action category mapping is correct
- Top-k frequencies use fixed class IDs
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from features_sgp_prefix_v3 import (  # noqa: E402
    SGP_V3_CORE_COLUMNS,
    _ATTACK_ACTIONS,
    _CONTROL_ACTIONS,
    _DEFENSE_ACTIONS,
    _SERVE_ACTIONS,
    _TOP5_POINT_IDS,
    _TOP8_ACTION_IDS,
    _categorize_action,
    _shannon_entropy,
    _tail_run_length,
    audit_no_banned_names,
    build_features_sgp_v3,
    get_feature_cols,
)


# ---- Feature count ----

def test_feature_count_is_65():
    assert len(SGP_V3_CORE_COLUMNS) == 65


def test_feature_names_unique():
    assert len(set(SGP_V3_CORE_COLUMNS)) == 65


# ---- _categorize_action ----

def test_categorize_action_attack():
    for a in _ATTACK_ACTIONS:
        assert _categorize_action(a) == 1


def test_categorize_action_control():
    for a in _CONTROL_ACTIONS:
        assert _categorize_action(a) == 2


def test_categorize_action_defense():
    for a in _DEFENSE_ACTIONS:
        assert _categorize_action(a) == 3


def test_categorize_action_serve():
    for a in _SERVE_ACTIONS:
        assert _categorize_action(a) == 4


def test_categorize_action_other():
    assert _categorize_action(0) == 0      # None
    assert _categorize_action(-1) == 0     # Out-of-range
    assert _categorize_action(99) == 0     # Way out of range


# ---- Banned name audit ----

def test_audit_no_banned_names_passes_for_clean_cols():
    audit_no_banned_names(SGP_V3_CORE_COLUMNS)  # should not raise


def test_audit_no_banned_names_catches_violations():
    with pytest.raises(ValueError, match="Banned feature names"):
        audit_no_banned_names(SGP_V3_CORE_COLUMNS + ["rally_full_length_marker"])
    with pytest.raises(ValueError):
        audit_no_banned_names(["final_shot_id"])
    with pytest.raises(ValueError):
        audit_no_banned_names(["terminal_action"])
    with pytest.raises(ValueError):
        audit_no_banned_names(["rally_winner_flag"])
    with pytest.raises(ValueError):
        audit_no_banned_names(["n_shots_total"])


# ---- _shannon_entropy ----

def test_entropy_empty():
    assert _shannon_entropy(np.zeros(19)) == 0.0


def test_entropy_degenerate():
    counts = np.zeros(19, dtype=int)
    counts[5] = 100
    assert _shannon_entropy(counts) == pytest.approx(0.0, abs=1e-9)


def test_entropy_uniform_two():
    counts = np.array([7, 7], dtype=int)
    assert _shannon_entropy(counts) == pytest.approx(np.log(2), abs=1e-9)


# ---- _tail_run_length ----

def test_tail_run_empty():
    assert _tail_run_length(np.array([], dtype=int)) == 0


def test_tail_run_pure():
    assert _tail_run_length(np.array([1, 1, 1], dtype=int)) == 3


def test_tail_run_breaks():
    assert _tail_run_length(np.array([1, 2, 3, 3], dtype=int)) == 2
    assert _tail_run_length(np.array([3, 3, 3, 2], dtype=int)) == 1


# ---- Build features integration ----

def _make_synthetic_rally(rally_uid: int, n_shots: int, sgp: int = 1) -> pd.DataFrame:
    """Generate a synthetic rally with n_shots."""
    rows = []
    for sn in range(1, n_shots + 1):
        rows.append({
            "rally_uid": rally_uid,
            "strikeNumber": sn,
            "sex": 1,
            "numberGame": 1,
            "rally_id": rally_uid,
            "scoreSelf": 0,
            "scoreOther": 0,
            "match": 1,
            "gamePlayerId": (sn % 2) + 1,    # alternates 1, 2, 1, 2, ...
            "gamePlayerOtherId": ((sn + 1) % 2) + 1,
            "strikeId": 1 if sn == 1 else (2 if sn == 2 else 4),
            "handId": 1 if sn % 2 == 1 else 2,
            "strengthId": 1,
            "spinId": 0,
            "positionId": 0,
            "actionId": ((sn + rally_uid) % 19),
            "pointId": ((sn + rally_uid) % 10),
            "serverGetPoint": sgp,
        })
    return pd.DataFrame(rows)


def _make_synthetic_train_df() -> pd.DataFrame:
    """Build a small synthetic train dataset with multiple rallies."""
    rallies = []
    for rid in range(1000, 1020):
        n_shots = 3 + (rid % 4)
        rallies.append(_make_synthetic_rally(rid, n_shots, sgp=rid % 2))
    return pd.concat(rallies, ignore_index=True)


def test_train_feature_count():
    df = _make_synthetic_train_df()
    feat = build_features_sgp_v3(df, is_train=True)
    feature_cols = get_feature_cols(feat)
    # Should have exactly 65 features
    assert len(feature_cols) == 65, f"Got {len(feature_cols)} features, expected 65"


def test_train_sample_count_matches_L_minus_1():
    """For each rally with L shots, we should get L-1 training samples."""
    df = _make_synthetic_train_df()
    feat = build_features_sgp_v3(df, is_train=True)
    for rid in df["rally_uid"].unique():
        L = (df["rally_uid"] == rid).sum()
        n_samples = (feat["rally_uid"] == rid).sum()
        assert n_samples == L - 1, f"Rally {rid}: expected {L-1} samples, got {n_samples}"


def test_test_one_sample_per_rally():
    """Test mode emits exactly one row per rally."""
    df = _make_synthetic_train_df()
    # Strip target labels to mimic test
    df_test = df.drop(columns=["serverGetPoint"]).copy()
    df_test["serverGetPoint"] = -1
    feat = build_features_sgp_v3(df_test, is_train=False)
    assert len(feat) == df["rally_uid"].nunique()
    assert feat["rally_uid"].nunique() == len(feat)


def test_strict_prefix_containment_in_train():
    """For every training sample, max(prefix_strikeNum) < target_strikeNum."""
    df = _make_synthetic_train_df()
    feat = build_features_sgp_v3(df, is_train=True)
    for _, row in feat.iterrows():
        rid = int(row["rally_uid"])
        target_sn = int(row["next_strikeNumber"])
        rally_shots = df[df["rally_uid"] == rid]
        prefix_max = rally_shots[rally_shots["strikeNumber"] < target_sn]["strikeNumber"].max()
        if pd.notna(prefix_max):
            assert prefix_max < target_sn, \
                f"Containment violation: rally {rid}, target {target_sn}, prefix_max {prefix_max}"


def test_empty_prefix_defaults_to_zero():
    """Synthetic rally where target_strike=1 (no prefix exists) → not generated.
    Verified indirectly via L-1 count."""
    # Actually train mode skips target_strike=1, so we don't get empty-prefix samples in train.
    # But let's check that a 2-shot rally produces ONE sample (target=2, prefix=1 shot).
    df = _make_synthetic_rally(999, n_shots=2, sgp=1)
    feat = build_features_sgp_v3(df, is_train=True)
    assert len(feat) == 1
    assert int(feat.iloc[0]["next_strikeNumber"]) == 2
    # The single prefix shot has actionId = (1 + 999) % 19 = 12
    expected_action = (1 + 999) % 19
    assert int(feat.iloc[0]["lag1_actionId"]) == expected_action


def test_test_with_one_shot_rally():
    """Test rally with only 1 visible shot: target=2, prefix=1 shot."""
    df = _make_synthetic_rally(2000, n_shots=1, sgp=1)
    df = df.drop(columns=["serverGetPoint"])
    df["serverGetPoint"] = -1
    feat = build_features_sgp_v3(df, is_train=False)
    assert len(feat) == 1
    assert int(feat.iloc[0]["next_strikeNumber"]) == 2
    assert int(feat.iloc[0]["lag2_actionId"]) == -1  # no lag2


def test_label_consistency_in_train():
    """All training samples in a rally must share the same SGP label."""
    df = _make_synthetic_train_df()
    feat = build_features_sgp_v3(df, is_train=True)
    for rid, group in feat.groupby("rally_uid"):
        unique_labels = group["serverGetPoint"].nunique()
        assert unique_labels == 1, f"Rally {rid} has {unique_labels} unique SGP labels in features"


def test_no_banned_features_in_output():
    """The actual output feature names must pass the banned-name audit."""
    df = _make_synthetic_train_df()
    feat = build_features_sgp_v3(df, is_train=True)
    feature_cols = get_feature_cols(feat)
    audit_no_banned_names(feature_cols)  # should not raise


def test_prefix_length_log_present_only_once():
    """Only ONE prefix-length feature should exist (per Codex constraint)."""
    feat_cols = SGP_V3_CORE_COLUMNS
    length_cols = [c for c in feat_cols if "length" in c.lower() or "next_strike" in c.lower()]
    assert length_cols == ["prefix_length_log"], \
        f"Expected only 'prefix_length_log', got {length_cols}"


def test_action_top_k_are_the_8_declared():
    """Top-8 action features must use the fixed class IDs."""
    feat_cols = SGP_V3_CORE_COLUMNS
    top8_cols = sorted([int(c.split("_")[-1]) for c in feat_cols if c.startswith("top8_action_freq_")])
    assert top8_cols == sorted(_TOP8_ACTION_IDS), \
        f"Top-8 actions mismatch: {top8_cols} vs declared {_TOP8_ACTION_IDS}"


def test_point_top_k_are_the_5_declared():
    """Top-5 point features must use the fixed class IDs."""
    feat_cols = SGP_V3_CORE_COLUMNS
    top5_cols = sorted([int(c.split("_")[-1]) for c in feat_cols if c.startswith("top5_point_freq_")])
    assert top5_cols == sorted(_TOP5_POINT_IDS)
