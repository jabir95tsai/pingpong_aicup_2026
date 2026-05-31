"""R-065c v2c consensus generator invariants.

Codex required (R-065b BLOCK 2026-05-23):
- No duplicate transformer votes
- Deterministic class cap by confidence ranking (not random)
- SGP sentinel for every pseudo row
- Versioned outputs separate from v2/v2b
- Generator tests cover: cap determinism, pool counts, sentinel guards, mask spec
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from build_pseudo_v2c_consensus import (  # noqa: E402
    GBM_CLUSTER,
    GBM_CLUSTER_TAG,
    TRANSFORMER_TEACHERS,
    ALL_TEACHERS,
    DEFAULT_THRESHOLDS,
    compute_consensus,
    deterministic_class_cap,
    slice_to_15,
)


# ─── Test 1: GBM cluster collapses to ONE vote ──────────────────────────────

def test_gbm_cluster_is_one_vote():
    """The GBM cluster must be represented as exactly one teacher in the pool,
    not as 4 individual teachers (Codex R-065b fix #1)."""
    assert GBM_CLUSTER_TAG in ALL_TEACHERS
    for member in GBM_CLUSTER:
        assert member not in ALL_TEACHERS, (
            f"Individual GBM member {member} should NOT be in ALL_TEACHERS "
            f"(it must vote only via the collapsed gbm_cluster)"
        )
    assert len(ALL_TEACHERS) == 1 + len(TRANSFORMER_TEACHERS)


# ─── Test 2: No duplicate transformer votes ─────────────────────────────────

def test_no_duplicate_transformer_votes():
    """Codex BLOCK fix #1: v11_aug_oldtest and v11_aug_oldtest_avg3 are
    numerically identical (max diff < 9e-08). Only one may appear in the pool."""
    forbidden_duplicates = [
        # (kept_tag, banned_duplicate_tag)
        ("v11_aug_oldtest", "v11_aug_oldtest_avg3"),
        ("v11plus_oldtest", "v11plus_oldtest_avg2"),
    ]
    for kept, dup in forbidden_duplicates:
        if kept in TRANSFORMER_TEACHERS:
            assert dup not in TRANSFORMER_TEACHERS, (
                f"{dup} is numerically identical to {kept}; "
                f"keeping both gives one model two votes."
            )


# ─── Test 3: Deterministic cap (re-run produces same kept row IDs) ──────────

def test_deterministic_cap_reproducible():
    """Cap selection must be deterministic — re-running with same input yields
    the same kept row IDs in the same order. No random subsampling."""
    rows = [
        {"rally_uid": 100, "pseudo_actionId": 1, "act_top1_p": 0.90, "act_sep": 0.5, "act_agree_count": 5},
        {"rally_uid": 200, "pseudo_actionId": 1, "act_top1_p": 0.85, "act_sep": 0.4, "act_agree_count": 5},
        {"rally_uid": 300, "pseudo_actionId": 1, "act_top1_p": 0.80, "act_sep": 0.3, "act_agree_count": 4},
        {"rally_uid": 400, "pseudo_actionId": 1, "act_top1_p": 0.75, "act_sep": 0.2, "act_agree_count": 4},
        {"rally_uid": 500, "pseudo_actionId": 2, "act_top1_p": 0.70, "act_sep": 0.2, "act_agree_count": 4},
        {"rally_uid": 600, "pseudo_actionId": 2, "act_top1_p": 0.65, "act_sep": 0.1, "act_agree_count": 4},
    ]
    # Pool size 6, cap 30% = 1 per class
    kept_a, dropped_a = deterministic_class_cap(
        rows,
        class_field="pseudo_actionId",
        rank_fields=["act_top1_p", "act_sep", "act_agree_count", "rally_uid"],
        cap_pct=0.30,
    )
    kept_b, dropped_b = deterministic_class_cap(
        list(rows),  # fresh copy of list
        class_field="pseudo_actionId",
        rank_fields=["act_top1_p", "act_sep", "act_agree_count", "rally_uid"],
        cap_pct=0.30,
    )
    assert [r["rally_uid"] for r in kept_a] == [r["rally_uid"] for r in kept_b]
    assert [r["rally_uid"] for r in dropped_a] == [r["rally_uid"] for r in dropped_b]


# ─── Test 4: Cap prefers highest top1_p, then sep, then agree, then rally_uid asc ──

def test_cap_ranking_order():
    """Highest top1_p wins; ties broken by sep, agree_count, then rally_uid (ASC)."""
    rows = [
        # Same class, same top1_p, different sep → higher sep wins
        {"rally_uid": 200, "pseudo_actionId": 7, "act_top1_p": 0.80, "act_sep": 0.10, "act_agree_count": 4},
        {"rally_uid": 100, "pseudo_actionId": 7, "act_top1_p": 0.80, "act_sep": 0.20, "act_agree_count": 4},
    ]
    # Cap = 1 → must keep row 100 (higher sep)
    kept, dropped = deterministic_class_cap(
        rows,
        class_field="pseudo_actionId",
        rank_fields=["act_top1_p", "act_sep", "act_agree_count", "rally_uid"],
        cap_pct=0.50,  # 50% of 2 = 1
    )
    assert len(kept) == 1
    assert kept[0]["rally_uid"] == 100
    assert dropped[0]["rally_uid"] == 200


# ─── Test 5: Cap rally_uid tie-break is ASCENDING ───────────────────────────

def test_cap_rally_uid_tiebreak_ascending():
    """When all rank fields tie, lower rally_uid wins (ASCENDING tie-break)."""
    rows = [
        {"rally_uid": 999, "pseudo_actionId": 5, "act_top1_p": 0.7, "act_sep": 0.1, "act_agree_count": 4},
        {"rally_uid": 100, "pseudo_actionId": 5, "act_top1_p": 0.7, "act_sep": 0.1, "act_agree_count": 4},
    ]
    kept, dropped = deterministic_class_cap(
        rows,
        class_field="pseudo_actionId",
        rank_fields=["act_top1_p", "act_sep", "act_agree_count", "rally_uid"],
        cap_pct=0.50,
    )
    assert kept[0]["rally_uid"] == 100, "Lower rally_uid should win the tie"


# ─── Test 6: Consensus enforces min_agree, top1_min, sep_min ────────────────

def test_consensus_threshold_enforcement():
    """A row that fails ANY threshold must not pass."""
    # 3-teacher test, 3 rows
    n_rows = 3
    # Build mock probs: row 0 — high consensus pass; row 1 — agree but low conf; row 2 — disagree
    probs_t1 = np.zeros((n_rows, 5))
    probs_t2 = np.zeros((n_rows, 5))
    probs_t3 = np.zeros((n_rows, 5))
    # Row 0: all 3 vote class 1 with high confidence
    probs_t1[0] = [0.05, 0.80, 0.10, 0.025, 0.025]
    probs_t2[0] = [0.05, 0.75, 0.10, 0.05,  0.05]
    probs_t3[0] = [0.05, 0.85, 0.05, 0.025, 0.025]
    # Row 1: all 3 vote class 2 but with LOW confidence (top1 ~0.4)
    probs_t1[1] = [0.30, 0.20, 0.40, 0.05, 0.05]
    probs_t2[1] = [0.30, 0.20, 0.40, 0.05, 0.05]
    probs_t3[1] = [0.30, 0.20, 0.40, 0.05, 0.05]
    # Row 2: t1+t2 vote class 3, t3 votes class 4 (2-of-3 agree)
    probs_t1[2] = [0.05, 0.05, 0.80, 0.05, 0.05]
    probs_t2[2] = [0.05, 0.05, 0.80, 0.05, 0.05]
    probs_t3[2] = [0.05, 0.05, 0.05, 0.80, 0.05]
    probs = {"t1": probs_t1, "t2": probs_t2, "t3": probs_t3}
    out = compute_consensus(
        probs,
        top1_min=0.60,  # row 1's 0.40 should fail
        sep_min=0.10,
        skip_classes=[],
        min_agree=3,    # row 2's 2-of-3 should fail
    )
    assert out["passed"][0], "Row 0 should pass (3/3 agree, high conf)"
    assert not out["passed"][1], "Row 1 should fail (top1=0.40 < 0.60)"
    assert not out["passed"][2], "Row 2 should fail (2/3 agree < min_agree=3)"


# ─── Test 7: skip_classes filter works ──────────────────────────────────────

def test_consensus_skip_classes():
    """Rows with consensus on a skip-class must not pass."""
    n_rows = 1
    probs = {}
    for t in ["t1", "t2", "t3"]:
        p = np.zeros((n_rows, 19))
        p[0, 15] = 0.9   # serve action
        p[0, 0] = 0.1
        probs[t] = p
    out = compute_consensus(
        probs,
        top1_min=0.5, sep_min=0.05,
        skip_classes=[15, 16, 17, 18],   # serves
        min_agree=3,
    )
    assert not out["passed"][0], "Serve-class consensus must be skipped"


# ─── Test 8: SGP sentinel guard in produced parquet ─────────────────────────

def test_parquet_sgp_sentinel():
    """Generated parquet must have serverGetPoint = -1 for every row.

    This test reads the actually-produced parquet from `data/pseudo_v2c.parquet`.
    """
    parquet_path = os.path.join(PROJECT_ROOT, "data", "pseudo_v2c.parquet")
    if not os.path.exists(parquet_path):
        pytest.skip(f"Parquet not yet generated: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    assert len(df) > 0, "Parquet should have at least 1 row"
    assert "serverGetPoint" in df.columns
    assert (df["serverGetPoint"] == -1).all(), "All rows must have SGP=-1 sentinel"


# ─── Test 9: Parquet schema has per-task masking columns ─────────────────────

def test_parquet_per_task_mask_columns():
    """Parquet schema must include kept_action and kept_point boolean columns
    so the trainer can apply per-task masking (Codex R-065b fix #4)."""
    parquet_path = os.path.join(PROJECT_ROOT, "data", "pseudo_v2c.parquet")
    if not os.path.exists(parquet_path):
        pytest.skip(f"Parquet not yet generated: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    for col in ["kept_action", "kept_point", "kept"]:
        assert col in df.columns
        assert df[col].dtype == bool, f"{col} should be boolean"
    # Every row must pass at least one task (i.e. kept_action OR kept_point)
    assert (df["kept_action"] | df["kept_point"]).all()


# ─── Test 10: Versioned outputs (no overwrite of v2/v2b artifacts) ───────────

def test_versioned_outputs_isolated():
    """`pseudo_v2c.parquet` must be distinct from `pseudo_v2.parquet` (no overwrite)."""
    v2_path = os.path.join(PROJECT_ROOT, "data", "pseudo_v2.parquet")
    v2c_path = os.path.join(PROJECT_ROOT, "data", "pseudo_v2c.parquet")
    if not (os.path.exists(v2_path) and os.path.exists(v2c_path)):
        pytest.skip("Either pseudo_v2.parquet or pseudo_v2c.parquet not present")
    assert os.path.abspath(v2_path) != os.path.abspath(v2c_path)
    df_v2 = pd.read_parquet(v2_path)
    df_v2c = pd.read_parquet(v2c_path)
    # They must differ (different teacher pool / thresholds = different rows)
    if len(df_v2) > 0 and len(df_v2c) > 0:
        s_v2 = set(df_v2["rally_uid"])
        s_v2c = set(df_v2c["rally_uid"])
        # We don't require full disjointness, but they shouldn't be identical
        assert not (s_v2 == s_v2c), "v2 and v2c kept sets identical — versioning is no-op"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
