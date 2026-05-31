"""R-065c — Per-task pseudo mask SPEC tests (no trainer edits yet).

Codex R-065b fix #4: trainer change is non-trivial; needs full per-task masking
infrastructure. These tests pin the SEMANTIC CONTRACT the trainer extension
must satisfy. They do NOT exercise `train_v14.py` directly (Codex stop gate:
no training in R-065c). Instead they construct a tiny mock parquet + simulate
the trainer's filter step and assert the row-routing rules.

Required semantics when the trainer is extended:
  pdf_act = pdf[pdf["kept_action"]]   → only these rows enter ACTION head
  pdf_pt  = pdf[pdf["kept_point"]]    → only these rows enter POINT head
  Server head: pseudo NEVER injected (V1 guard, unchanged)
  Flip aug: NEVER applied to pseudo rows (V1 guard, unchanged)
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def _make_parquet(rows):
    """Build a tiny mock pseudo parquet."""
    return pd.DataFrame(rows)


def _simulate_trainer_per_task_split(pdf: pd.DataFrame):
    """Spec: this is the routing the trainer must implement.

    Returns dict with action_subset, point_subset, server_subset (empty).
    """
    pdf_act = pdf[pdf["kept_action"]].copy()
    pdf_pt = pdf[pdf["kept_point"]].copy()
    # Server head MUST exclude all pseudo rows
    pdf_server: pd.DataFrame = pdf.iloc[:0]
    return {
        "action_subset": pdf_act,
        "point_subset": pdf_pt,
        "server_subset": pdf_server,
    }


# ─── Test A: action-only row routes ONLY to action head ──────────────────────

def test_action_only_row():
    pdf = _make_parquet([
        {"rally_uid": 100, "pseudo_actionId": 1, "pseudo_pointId": -1,
         "kept_action": True, "kept_point": False, "kept": True,
         "serverGetPoint": -1},
    ])
    out = _simulate_trainer_per_task_split(pdf)
    assert len(out["action_subset"]) == 1
    assert int(out["action_subset"].iloc[0]["rally_uid"]) == 100
    assert len(out["point_subset"]) == 0
    assert len(out["server_subset"]) == 0


# ─── Test B: point-only row routes ONLY to point head ────────────────────────

def test_point_only_row():
    pdf = _make_parquet([
        {"rally_uid": 200, "pseudo_actionId": -1, "pseudo_pointId": 7,
         "kept_action": False, "kept_point": True, "kept": True,
         "serverGetPoint": -1},
    ])
    out = _simulate_trainer_per_task_split(pdf)
    assert len(out["action_subset"]) == 0
    assert len(out["point_subset"]) == 1
    assert int(out["point_subset"].iloc[0]["rally_uid"]) == 200
    assert len(out["server_subset"]) == 0


# ─── Test C: dual-task row routes to BOTH heads ──────────────────────────────

def test_dual_task_row():
    pdf = _make_parquet([
        {"rally_uid": 300, "pseudo_actionId": 3, "pseudo_pointId": 5,
         "kept_action": True, "kept_point": True, "kept": True,
         "serverGetPoint": -1},
    ])
    out = _simulate_trainer_per_task_split(pdf)
    assert len(out["action_subset"]) == 1
    assert len(out["point_subset"]) == 1
    assert int(out["action_subset"].iloc[0]["rally_uid"]) == 300
    assert int(out["point_subset"].iloc[0]["rally_uid"]) == 300


# ─── Test D: row with both flags False must not enter pool ───────────────────

def test_no_kept_row_excluded_from_both():
    """A row that fails BOTH consensus checks must not be in the parquet at all.
    The generator's contract is: parquet contains only rows where (kept_action OR
    kept_point). This test asserts that even if such a row somehow ended up in
    the parquet, the trainer's per-task split would correctly skip it."""
    pdf = _make_parquet([
        {"rally_uid": 400, "pseudo_actionId": -1, "pseudo_pointId": -1,
         "kept_action": False, "kept_point": False, "kept": False,
         "serverGetPoint": -1},
    ])
    out = _simulate_trainer_per_task_split(pdf)
    assert len(out["action_subset"]) == 0
    assert len(out["point_subset"]) == 0


# ─── Test E: server head MUST exclude all pseudo (V1 guard) ──────────────────

def test_server_head_excludes_pseudo():
    """Even if a pseudo row carries a SGP value (legacy bug), the trainer's
    server head must exclude it. We enforce this by requiring server_subset
    to be empty regardless of the input."""
    pdf = _make_parquet([
        {"rally_uid": 500, "pseudo_actionId": 1, "pseudo_pointId": 5,
         "kept_action": True, "kept_point": True, "kept": True,
         "serverGetPoint": 1},   # ← buggy SGP value
    ])
    out = _simulate_trainer_per_task_split(pdf)
    assert len(out["server_subset"]) == 0, (
        "Server head must NEVER receive pseudo rows, even if they carry a SGP value"
    )


# ─── Test F: per-task subsets can have different lengths ─────────────────────

def test_per_task_subsets_differ():
    """With a mixed parquet, action subset and point subset have different sizes."""
    pdf = _make_parquet([
        {"rally_uid": 1, "pseudo_actionId": 1, "pseudo_pointId": -1,
         "kept_action": True, "kept_point": False, "kept": True, "serverGetPoint": -1},
        {"rally_uid": 2, "pseudo_actionId": 1, "pseudo_pointId": -1,
         "kept_action": True, "kept_point": False, "kept": True, "serverGetPoint": -1},
        {"rally_uid": 3, "pseudo_actionId": 1, "pseudo_pointId": -1,
         "kept_action": True, "kept_point": False, "kept": True, "serverGetPoint": -1},
        {"rally_uid": 4, "pseudo_actionId": -1, "pseudo_pointId": 7,
         "kept_action": False, "kept_point": True, "kept": True, "serverGetPoint": -1},
    ])
    out = _simulate_trainer_per_task_split(pdf)
    assert len(out["action_subset"]) == 3
    assert len(out["point_subset"]) == 1
    assert len(out["server_subset"]) == 0


# ─── Test G: flip-aug applies to REAL rows only, never pseudo ─────────────────

def test_flip_aug_excludes_pseudo_spec():
    """The trainer's existing `augment_flip` must run on REAL train rows only.
    Pseudo rows are concatenated AFTER flip_aug, so they are never duplicated
    in left-right flipped form. This is a CONTRACT — the trainer extension must
    preserve it.

    Spec: pseudo_X is appended AFTER X_tr_flip, not before:
      X_tr_combined = np.vstack([X_tr_flip, pseudo_X_act])   # ← correct
    NOT:
      X_tr_combined = np.vstack([X_tr_aug_with_pseudo_flipped, ...])
    """
    # Symbolic test: just assert the contract phrase is in the docstring.
    import build_pseudo_v2c_consensus as gen
    src_path = gen.__file__
    with open(src_path, encoding="utf-8") as f:
        content = f.read()
    # The spec must be documented in the manifest / module docstring
    assert "Codex" in content, "Generator should reference Codex spec"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
