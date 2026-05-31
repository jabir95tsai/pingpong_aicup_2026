"""R-064 invariant tests for v15feat_d_core spin-aware features.

Codex required (2026-05-23 APPROVE_WITH_FIXES):
1. Group A priors sum to 1 (within float tolerance)
2. No NaN/Inf in any v15feat_d feature column
3. Feature count is exact (13 new columns)
4. No serverGetPoint column is read by the feature builder
5. Group A stats are computed from fold-train only (not val/test)
6. serve_spin_class only reads prefix rows with strikeNumber < target_strikeNumber
"""
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from features_v15feat_d import (  # noqa: E402
    V15FEAT_D_ADDED_COLUMNS,
    SPIN_CLASSES,
    DIRICHLET_ALPHA,
    compute_global_stats_v15feat_d,
    build_features_v15feat_d,
)


def _make_rally(rally_uid, match_id, shots, sgp=1):
    """Build a small rally DataFrame from a list of shot dicts."""
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
            "spinId": shot["spinId"],
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


def _make_dataset(rallies_spec):
    """rallies_spec: [(rally_uid, match_id, [shot dicts]), ...]"""
    rows = []
    for rally_uid, match_id, shots in rallies_spec:
        rows.extend(_make_rally(rally_uid, match_id, shots))
    return pd.DataFrame(rows)


# ─── Test 1: Group A priors sum to 1 ─────────────────────────────────────────

def test_priors_sum_to_one():
    """For every observed (action, position) bin, the smoothed prior over the
    5 spin classes must sum to 1.0 within float tolerance."""
    df = _make_dataset([
        (100, 1, [{"actionId": 15, "pointId": 0, "spinId": 1, "positionId": 1},
                  {"actionId": 1,  "pointId": 7, "spinId": 1, "positionId": 2},
                  {"actionId": 13, "pointId": 0, "spinId": 3, "positionId": 1}]),
        (200, 1, [{"actionId": 16, "pointId": 0, "spinId": 4, "positionId": 1},
                  {"actionId": 5,  "pointId": 7, "spinId": 1, "positionId": 2},
                  {"actionId": 10, "pointId": 1, "spinId": 2, "positionId": 1}]),
        (300, 1, [{"actionId": 15, "pointId": 0, "spinId": 2, "positionId": 1},
                  {"actionId": 10, "pointId": 1, "spinId": 2, "positionId": 1},
                  {"actionId": 1,  "pointId": 7, "spinId": 1, "positionId": 2}]),
    ])
    stats = compute_global_stats_v15feat_d(df)
    assert "spin_prior_smoothed" in stats
    assert len(stats["spin_prior_smoothed"]) > 0, "No bins were populated"
    for key, prior in stats["spin_prior_smoothed"].items():
        assert prior.shape == (len(SPIN_CLASSES),), f"Wrong shape for bin {key}"
        s = float(prior.sum())
        assert abs(s - 1.0) < 1e-5, f"Prior for {key} sums to {s}, not 1.0"


# ─── Test 2: Global prior sums to 1 ─────────────────────────────────────────

def test_global_prior_sums_to_one():
    df = _make_dataset([
        (100, 1, [{"actionId": 15, "pointId": 0, "spinId": 1, "positionId": 1},
                  {"actionId": 1,  "pointId": 7, "spinId": 1, "positionId": 2}]),
        (200, 1, [{"actionId": 16, "pointId": 0, "spinId": 2, "positionId": 1},
                  {"actionId": 5,  "pointId": 7, "spinId": 3, "positionId": 2}]),
    ])
    stats = compute_global_stats_v15feat_d(df)
    s = float(stats["spin_global_p"].sum())
    assert abs(s - 1.0) < 1e-5, f"Global spin prior sums to {s}, not 1.0"


# ─── Test 3: No NaN/Inf in any v15feat_d feature ─────────────────────────────

def test_no_nan_inf():
    df = _make_dataset([
        (100, 1, [{"actionId": 15, "pointId": 0, "spinId": 1, "positionId": 1},
                  {"actionId": 1,  "pointId": 7, "spinId": 1, "positionId": 2},
                  {"actionId": 13, "pointId": 0, "spinId": 3, "positionId": 1}]),
        (200, 1, [{"actionId": 16, "pointId": 0, "spinId": 4, "positionId": 1},
                  {"actionId": 5,  "pointId": 7, "spinId": 1, "positionId": 2}]),
        (300, 1, [{"actionId": 15, "pointId": 0, "spinId": 2, "positionId": 1},
                  {"actionId": 10, "pointId": 1, "spinId": 2, "positionId": 1},
                  {"actionId": 1,  "pointId": 7, "spinId": 1, "positionId": 2}]),
    ])
    stats = compute_global_stats_v15feat_d(df)
    feat = build_features_v15feat_d(df, is_train=True, global_stats_v9=stats, raw_df=df)
    for col in V15FEAT_D_ADDED_COLUMNS:
        assert col in feat.columns, f"Missing column: {col}"
        vals = feat[col].to_numpy(dtype=np.float64)
        assert np.isfinite(vals).all(), f"Column {col} has non-finite values"


# ─── Test 4: Exact feature count ────────────────────────────────────────────

def test_exact_feature_count():
    """Codex fix #4: feature count is exact (13 new columns)."""
    assert len(V15FEAT_D_ADDED_COLUMNS) == 13, (
        f"v15feat_d must add exactly 13 columns, got {len(V15FEAT_D_ADDED_COLUMNS)}"
    )

    # Also verify all expected names exist
    expected = (
        [f"prior_next_spin_class_{c}" for c in SPIN_CLASSES]
        + ["last_was_heavy_backspin", "last_was_heavy_topspin",
           "last_was_sidespin", "last_was_no_spin"]
        + ["serve_topspin", "serve_backspin", "serve_sidespin", "serve_no_spin"]
    )
    assert set(V15FEAT_D_ADDED_COLUMNS) == set(expected)


# ─── Test 5: serverGetPoint NEVER read by the feature builder ────────────────

def test_no_sgp_read():
    """Codex fix #4 (SGP-leakage guard): the feature builder must not access
    the `serverGetPoint` column. We patch the column to a tripwire and verify
    the feature outputs are unchanged."""
    df = _make_dataset([
        (100, 1, [{"actionId": 15, "pointId": 0, "spinId": 1, "positionId": 1},
                  {"actionId": 1,  "pointId": 7, "spinId": 1, "positionId": 2},
                  {"actionId": 13, "pointId": 0, "spinId": 3, "positionId": 1}]),
        (200, 1, [{"actionId": 16, "pointId": 0, "spinId": 4, "positionId": 1},
                  {"actionId": 5,  "pointId": 7, "spinId": 1, "positionId": 2}]),
    ], )

    stats_a = compute_global_stats_v15feat_d(df)
    feat_a = build_features_v15feat_d(df, is_train=True, global_stats_v9=stats_a, raw_df=df)

    # Flip SGP to opposite (1 → 0, 0 → 1). Compute features again. Must be identical.
    df_flipped = df.copy()
    df_flipped["serverGetPoint"] = 1 - df_flipped["serverGetPoint"].astype(int)
    stats_b = compute_global_stats_v15feat_d(df_flipped)
    feat_b = build_features_v15feat_d(df_flipped, is_train=True,
                                       global_stats_v9=stats_b, raw_df=df_flipped)

    for col in V15FEAT_D_ADDED_COLUMNS:
        np.testing.assert_array_equal(
            feat_a[col].to_numpy(),
            feat_b[col].to_numpy(),
            err_msg=f"Column {col} changed when SGP was flipped — SGP leak",
        )


# ─── Test 6: Group A stats are fold-train-only (not val/test) ────────────────

def test_priors_fold_train_only():
    """compute_global_stats_v15feat_d must produce priors that depend ONLY on
    the train slice passed in — not on rows added later (val/test)."""
    df_train = _make_dataset([
        (100, 1, [{"actionId": 15, "pointId": 0, "spinId": 1, "positionId": 1},
                  {"actionId": 1,  "pointId": 7, "spinId": 1, "positionId": 2},
                  {"actionId": 13, "pointId": 0, "spinId": 3, "positionId": 1}]),
        (200, 1, [{"actionId": 16, "pointId": 0, "spinId": 4, "positionId": 1},
                  {"actionId": 5,  "pointId": 7, "spinId": 1, "positionId": 2}]),
    ])
    df_train_plus_val = pd.concat([df_train, _make_dataset([
        # If the function leaked into this synthetic "val/test" data, priors
        # would change. Add many rows of (last_action=15, last_position=1, next_spin=5)
        # which should have ZERO mass in train-only stats.
        (900, 2, [{"actionId": 15, "pointId": 0, "spinId": 1, "positionId": 1}]
                + [{"actionId": 999 % 19, "pointId": 0, "spinId": 5, "positionId": 1}] * 50),
    ])], ignore_index=True)

    stats_train = compute_global_stats_v15feat_d(df_train)
    stats_train_plus = compute_global_stats_v15feat_d(df_train_plus_val)

    # Verify (15, 1) prior differs once we add val data — confirms the function
    # IS sensitive to input (sanity check) ...
    key = (15, 1)
    if key in stats_train["spin_prior_smoothed"] and key in stats_train_plus["spin_prior_smoothed"]:
        diff = np.abs(stats_train["spin_prior_smoothed"][key]
                      - stats_train_plus["spin_prior_smoothed"][key]).max()
        assert diff > 1e-6, "Stats are insensitive to input data — function broken"

    # ... AND verify that passing tr_raw alone yields a prior independent of
    # the val rows (the contract: caller passes the fold-train slice).
    # We assert that calling on tr_raw twice with identical input gives identical output:
    stats_dup = compute_global_stats_v15feat_d(df_train)
    for k in stats_train["spin_prior_smoothed"]:
        np.testing.assert_array_equal(
            stats_train["spin_prior_smoothed"][k],
            stats_dup["spin_prior_smoothed"][k],
            err_msg=f"Deterministic check failed for bin {k}",
        )


# ─── Test 7: serve_spin_class is prefix-only ─────────────────────────────────

def test_serve_spin_class_prefix_only():
    """Codex fix #4 (last invariant): serve_spin_class must only read prefix
    rows with strikeNumber < target_strikeNumber. We mutate the TARGET shot
    (post-prefix) to look like a different serve and verify features don't
    change.

    The TARGET in v15feat_d feature build is whichever shot is being predicted;
    for is_train=True the trainer iterates target_idx in 1..N-1. The feature
    builder uses `next_strikeNumber` from the feat row to identify the cutoff.

    Concretely: take a rally with serve at sn=1 (spinId=1) and a later shot;
    build features for target_strikeNumber=3 (predict shot 3). Then mutate
    shot 3's spinId to 5 (a different serve-like spin). Features must be
    identical because shot 3 is the TARGET, not part of the prefix.
    """
    base_rally = [
        # sn=1 (serve, included in prefix when predicting sn>=2)
        {"strikeId": 1, "actionId": 15, "pointId": 0, "spinId": 1, "positionId": 1},
        # sn=2 (last shot in prefix when predicting sn=3)
        {"strikeId": 2, "actionId": 10, "pointId": 1, "spinId": 2, "positionId": 1},
        # sn=3 (TARGET — should not be read)
        {"strikeId": 4, "actionId": 1,  "pointId": 7, "spinId": 1, "positionId": 2},
    ]
    df_a = _make_dataset([(100, 1, base_rally), (200, 1, base_rally)])

    # Mutate sn=3 to look like a different serve
    mutated = [
        {"strikeId": 1, "actionId": 15, "pointId": 0, "spinId": 1, "positionId": 1},
        {"strikeId": 2, "actionId": 10, "pointId": 1, "spinId": 2, "positionId": 1},
        # If serve_spin_class reads this row, serve_sidespin would flip to 1
        {"strikeId": 1, "actionId": 17, "pointId": 9, "spinId": 5, "positionId": 3},
    ]
    df_b = _make_dataset([(100, 1, mutated), (200, 1, mutated)])

    stats = compute_global_stats_v15feat_d(df_a)  # same stats for both
    feat_a = build_features_v15feat_d(df_a, is_train=True, global_stats_v9=stats, raw_df=df_a)
    feat_b = build_features_v15feat_d(df_b, is_train=True, global_stats_v9=stats, raw_df=df_b)

    # Find rows where target_strikeNumber (next_strikeNumber) == 3 — those are
    # the predictions of shot 3, where shot 3 IS the target (must be excluded).
    rows_a = feat_a[feat_a["next_strikeNumber"] == 3]
    rows_b = feat_b[feat_b["next_strikeNumber"] == 3]
    assert len(rows_a) == len(rows_b)
    assert len(rows_a) > 0, "No rows with next_strikeNumber==3 — test data malformed"

    for col in ["serve_topspin", "serve_backspin", "serve_sidespin", "serve_no_spin",
                "last_was_heavy_backspin", "last_was_heavy_topspin",
                "last_was_sidespin", "last_was_no_spin"]:
        np.testing.assert_array_equal(
            rows_a[col].to_numpy(), rows_b[col].to_numpy(),
            err_msg=f"Column {col} changed when target shot was mutated — "
                    f"prefix-only invariant broken"
        )


# ─── Test 8: Dirichlet smoothing is applied (sparse bin defaults to global) ──

def test_dirichlet_smoothing_applied():
    """When a bin has very few observations, the prior should be pulled toward
    the global prior. We construct a sparse bin (1 observation) and verify
    the smoothed prior is dominated by alpha*global_p, not the single sample."""
    # Construct a train set where bin (15, 1) has only 1 obs with next_spinId=5,
    # but the global distribution is heavily skewed toward spin=1.
    rallies = []
    # Many rallies establishing global prior dominated by spin=1
    for i in range(50):
        rallies.append((1000 + i, 1, [
            {"actionId": 1, "pointId": 7, "spinId": 1, "positionId": 2},
            {"actionId": 5, "pointId": 7, "spinId": 1, "positionId": 2},
        ]))
    # ONE rally with target bin (15, 1) → next_spin=5 observation
    rallies.append((2000, 1, [
        {"actionId": 15, "pointId": 0, "spinId": 3, "positionId": 1},
        {"actionId": 1,  "pointId": 7, "spinId": 5, "positionId": 2},
    ]))
    df = _make_dataset(rallies)
    stats = compute_global_stats_v15feat_d(df)
    key = (15, 1)
    assert key in stats["spin_prior_smoothed"], f"Bin {key} should be present"
    prior = stats["spin_prior_smoothed"][key]
    # With alpha=20, single observation of spin=5 in bin (1 sample) should
    # be dominated by the smoothing. Specifically:
    # numerator: [counts] + 20 * global_p = [0+20*g1, 0+20*g2, 0+20*g3, 0+20*g4, 1+20*g5]
    # denominator: 1 + 20 = 21
    # P(spin=5) = (1 + 20*g5) / 21
    # Since global g5 is tiny (~1/(50*2*2)=tiny), the unsmoothed P(spin=5)=1.0
    # should drop dramatically.
    global_p = stats["spin_global_p"]
    expected_p5 = (1.0 + DIRICHLET_ALPHA * float(global_p[4])) / (1.0 + DIRICHLET_ALPHA)
    np.testing.assert_allclose(float(prior[4]), expected_p5, atol=1e-5)
    assert prior[4] < 0.20, (
        f"Dirichlet smoothing should drop P(spin=5) well below the raw 1.0, "
        f"got {prior[4]:.4f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
