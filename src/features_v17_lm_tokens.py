"""features_v17_lm_tokens — token sequence builder for v17_causal_lm (R-013).

Builds shot-level token arrays for the causal autoregressive rally LM.
Shared across Phase 1a (test visible prefixes), Phase 1b (Fold-1 train
rallies), and Phase 2 (standard supervised pairs).

Design notes (per R-013 / Codex APPROVE_WITH_FIXES):
- One position per shot. No EOS. No BOS for smoke (kept the API simple).
  The causal mask in the model handles "predict next" semantics.
- Per-shot token = factored embedding sum of:
    * actionId      (15 + PAD; 15-18 mapped to 0 to match v11/v14 supervised)
    * pointId       (10 + PAD)
    * handId        (3 + PAD)
    * strengthId    (4 + PAD)
    * spinId        (6 + PAD)
    * positionId    (4 + PAD)
    * strikeId      (5 + PAD)  (already remapped via STRIKE_ID_MAP)
    * shooter_side  (2 + PAD)  (server-side 0 / returner-side 1, alternating)
- Per-rally meta added as a context bias (sex, numberGame) — NOT a token.
- LEAKAGE INVARIANTS (Codex fix #8 enforced in audit functions below):
    1. NO serverGetPoint anywhere (not in token vocabulary, not in fields).
    2. NO match identifier (dataset metadata).
    3. NO rally_uid (dataset metadata).
    4. NO gamePlayerId / gamePlayerOtherId (de-identified; would not transfer).
    5. shooter_side carries server/returner role only — derived from strikeNumber
       parity within the rally, NOT from match-side / camera-side / player-id.

Forbidden field set is asserted at build time; see audit_no_forbidden_fields.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── Token field registry ────────────────────────────────────────────────────
# (field_name, n_classes_excluding_PAD, PAD_index)
# PAD index is appended as the LAST class; e.g. action has classes 0..14, PAD=15.
TOKEN_FIELDS = [
    ("actionId",   15, 15),
    ("pointId",    10, 10),
    ("handId",      3,  3),
    ("strengthId", 4,  4),
    ("spinId",     6,  6),
    ("positionId", 4,  4),
    ("strikeId",   5,  5),  # already 0..4 via STRIKE_ID_MAP
    ("shooter_side", 2, 2),
]
TOKEN_FIELD_NAMES = [f[0] for f in TOKEN_FIELDS]
TOKEN_PAD_INDEX = {f[0]: f[2] for f in TOKEN_FIELDS}
TOKEN_VOCAB_SIZES = {f[0]: f[1] + 1 for f in TOKEN_FIELDS}  # +1 for PAD

META_FIELDS = ["sex", "numberGame"]   # NOT a token; carried as numerical context
META_DIM = len(META_FIELDS)

# Fields the v17 LM is FORBIDDEN to read (Codex fix #8 audit).
FORBIDDEN_FIELDS = {
    "serverGetPoint",
    "match",
    "rally_uid",
    "gamePlayerId",
    "gamePlayerOtherId",
}


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class RallySequence:
    """Per-rally token sequence used by both Phase 1 (next-token) and
    Phase 2 (supervised) datasets."""
    rally_uid: str       # used ONLY for Phase 1 corpus filtering / audit;
                         # NEVER an embedding input
    match_id: str        # used ONLY for fold split; NEVER an embedding input
    shots: np.ndarray    # int8 (T, 8)  — columns in TOKEN_FIELD_NAMES order
    meta: np.ndarray     # float32 (META_DIM,) — sex, numberGame normalised
    sgp: int             # rally-level SGP label (-1 if test rally)
    full_length: int     # number of shots in this rally


# ─── Per-rally builder ───────────────────────────────────────────────────────

def _shooter_side_array(strike_numbers: np.ndarray) -> np.ndarray:
    """0 for server side (odd strikeNumber 1, 3, 5, ...), 1 for returner side
    (even strikeNumber 2, 4, ...)."""
    return ((strike_numbers % 2) == 0).astype(np.int8)


def build_rally_sequence(rally_grp: pd.DataFrame, is_test: bool) -> RallySequence:
    """Build one RallySequence from a per-rally dataframe.

    For TRAIN rallies, full sequence (1..T) is produced.
    For TEST rallies, the visible prefix (1..L where L = visible length) is
    produced; no hidden target.
    """
    grp = rally_grp.sort_values("strikeNumber").reset_index(drop=True)
    T = len(grp)
    if T == 0:
        raise ValueError(f"rally has zero shots")

    # Per-shot token columns
    sn = grp["strikeNumber"].to_numpy(dtype=np.int32)
    action_raw = grp["actionId"].to_numpy(dtype=np.int32)
    # Map action 15-18 → 0 to match v11/v14 supervised target distribution.
    # Serves are dominated by class 15-18 at strikeNumber=1 only; these
    # never appear as supervised next-shot targets in v11/v14, so we
    # collapse them to class 0 ("None/other") here for consistent
    # next-token modelling.
    action = np.where(action_raw >= 15, 0, action_raw).astype(np.int8)

    point      = grp["pointId"].to_numpy(dtype=np.int8)
    hand       = grp["handId"].to_numpy(dtype=np.int8)
    strength   = grp["strengthId"].to_numpy(dtype=np.int8)
    spin       = grp["spinId"].to_numpy(dtype=np.int8)
    position   = grp["positionId"].to_numpy(dtype=np.int8)
    strike_id  = grp["strikeId"].to_numpy(dtype=np.int8)  # already 0..4
    side       = _shooter_side_array(sn)

    shots = np.stack([action, point, hand, strength, spin, position,
                      strike_id, side], axis=1).astype(np.int8)
    assert shots.shape == (T, 8), f"shape {shots.shape} != ({T}, 8)"

    # Meta context (rally-level, not a token)
    sex = float(grp["sex"].iloc[0]) / 2.0
    num_game = float(grp["numberGame"].iloc[0]) / 7.0
    meta = np.array([sex, num_game], dtype=np.float32)

    # Rally-level SGP label
    if is_test or "serverGetPoint" not in grp.columns:
        sgp_label = -1
    else:
        sgp_label = int(grp["serverGetPoint"].iloc[0])

    rally_uid = str(grp["rally_uid"].iloc[0])
    match_id = str(grp["match"].iloc[0])

    return RallySequence(rally_uid=rally_uid, match_id=match_id,
                         shots=shots, meta=meta, sgp=sgp_label,
                         full_length=T)


def build_phase1_corpus(raw_df: pd.DataFrame, rally_filter: set | None,
                        is_test: bool, label: str) -> list[RallySequence]:
    """Build Phase 1 next-token pretraining corpus.

    Phase 1a (test visible prefixes): pass `is_test=True`, `rally_filter=None`.
    Phase 1b (Fold-1 train rallies):  pass `is_test=False`, `rally_filter=set(fold1_train_rally_uids)`.

    `rally_filter` is the set of rally_uids to KEEP. None = keep all.
    """
    seqs: list[RallySequence] = []
    n_skipped = 0
    for uid, grp in raw_df.groupby("rally_uid", sort=False):
        if rally_filter is not None and str(uid) not in rally_filter:
            n_skipped += 1
            continue
        seq = build_rally_sequence(grp, is_test=is_test)
        if seq.full_length < 2:
            # Need at least 2 shots to train predict-next on position 0
            # (predicting shot 2 from shot 1 representation).
            continue
        seqs.append(seq)
    print(f"  [v17_tokens] Phase1 corpus '{label}': {len(seqs)} rallies "
          f"(filtered out {n_skipped})")
    return seqs


# ─── Audit functions (Codex fix #8) ──────────────────────────────────────────

def audit_no_forbidden_fields(token_field_names: list[str]) -> dict:
    """8.D part 1 — token builder must not read forbidden fields.

    Run BEFORE training. Aborts on violation."""
    used = set(token_field_names)
    intersection = used & FORBIDDEN_FIELDS
    assert not intersection, (
        f"VIOLATION (8.D): token fields contain forbidden fields: {intersection}")
    # Plus the immutable check that META_FIELDS does not contain anything bad
    intersection_meta = set(META_FIELDS) & FORBIDDEN_FIELDS
    assert not intersection_meta, (
        f"VIOLATION (8.D): meta fields contain forbidden fields: {intersection_meta}")
    return {"audit_8D_token_fields": "PASS",
            "token_fields": list(used),
            "meta_fields": list(META_FIELDS),
            "forbidden_set": sorted(FORBIDDEN_FIELDS)}


def audit_fold_safe_pretrain(phase1a_rally_uids: set, phase1b_rally_uids: set,
                              fold1_train_rally_uids: set,
                              fold1_val_rally_uids: set) -> dict:
    """8.A — Phase 1 (a + b) corpus must be disjoint from Fold-1 val rallies."""
    phase1_total = phase1a_rally_uids | phase1b_rally_uids
    overlap = phase1_total & fold1_val_rally_uids
    assert not overlap, (
        f"VIOLATION (8.A): Phase 1 corpus overlaps Fold-1 val rallies "
        f"({len(overlap)} rallies; sample: {list(overlap)[:5]})")

    # Phase 1b (the train continuation) must equal Fold-1 train rally set
    # (intersection of train rallies and rallies present in our corpus).
    missing = fold1_train_rally_uids - phase1b_rally_uids
    extra = phase1b_rally_uids - fold1_train_rally_uids
    # Allow some rallies to be skipped (e.g. <2 shots), but fail on extras.
    assert not extra, (
        f"VIOLATION (8.A): Phase 1b corpus contains rallies NOT in "
        f"Fold-1 train ({len(extra)}; sample {list(extra)[:5]})")
    return {
        "audit_8A_fold_safe": "PASS",
        "phase1a_rallies": len(phase1a_rally_uids),
        "phase1b_rallies": len(phase1b_rally_uids),
        "fold1_train_rallies": len(fold1_train_rally_uids),
        "fold1_val_rallies": len(fold1_val_rally_uids),
        "phase1b_missing_from_train_(skipped_e.g._<2_shots)": len(missing),
    }


def audit_test_prefix_length(test_corpus: list[RallySequence],
                              test_df: pd.DataFrame) -> dict:
    """8.C — for each test rally, Phase 1a sequence length equals visible length.
    No hidden target appended."""
    # Normalise keys to str to match RallySequence.rally_uid storage.
    test_visible_lengths = {str(k): int(v) for k, v in
                            test_df.groupby("rally_uid")["strikeNumber"]
                                  .max().to_dict().items()}
    n_violations = 0
    samples = []
    for seq in test_corpus[:50]:
        expected = test_visible_lengths[str(seq.rally_uid)]
        actual = seq.full_length
        ok = (actual == expected)
        if len(samples) < 10:
            samples.append({"rally_uid": str(seq.rally_uid)[:32],
                            "expected": expected, "actual": actual, "ok": ok})
        if not ok:
            n_violations += 1
    # Hard assertion across the full corpus (not just first 50)
    for seq in test_corpus:
        expected = test_visible_lengths[str(seq.rally_uid)]
        assert seq.full_length == expected, (
            f"VIOLATION (8.C): test rally {str(seq.rally_uid)[:16]} Phase-1a "
            f"length {seq.full_length} != visible {expected}")
    return {
        "audit_8C_test_prefix_length": "PASS",
        "samples": samples,
        "n_test_rallies_checked": len(test_corpus),
        "violations_in_first_50_sample": n_violations,
    }


def audit_no_target_in_prefix(supervised_dataset, max_check: int = 1000) -> dict:
    """8.B — for every supervised pair, input sequence has length exactly
    N-1 and target shot N is NOT in input."""
    n_checked = 0
    samples = []
    for s in supervised_dataset[:max_check]:
        N = int(s["next_sn"])
        # v11 build_samples puts shots 0..N-1 (Python slice [:N], length N) into
        # cat_seq when target is at strikeNumber N (1-indexed). Since
        # strikeNumber is 1-indexed and grp.sort_values("strikeNumber") then
        # cat_seq[:tgt] for tgt=k uses indices 0..k-1 corresponding to shots
        # at strikeNumber 1..k, then target is at index k = strikeNumber k+1.
        # So for next_sn = N, the input has N-1 shots (shots 1..N-1).
        cat_seq = s["cat_seq"]
        T_input = cat_seq.shape[0]
        assert T_input == N - 1, (
            f"VIOLATION (8.B): sample next_sn={N} has input length {T_input} "
            f"(expected N-1 = {N-1})")
        # No row in cat_seq should be at strikeNumber >= N. Since cat_seq does
        # not store strikeNumber directly, we check by length only.
        # (build_samples iterates strikeNumber 1..N-1 to fill cat_seq[:N-1]
        #  per its tgt loop, so length check is sufficient.)
        n_checked += 1
        if len(samples) < 10:
            samples.append({"next_sn": N, "input_length": T_input, "ok": True})
    return {
        "audit_8B_no_target_in_prefix": "PASS",
        "n_supervised_checked": n_checked,
        "samples": samples,
    }


def audit_sgp_loss_count(phase1_sgp_count: int,
                         phase2_sgp_count: int,
                         phase2_train_rally_count: int,
                         phase2_test_sgp_count: int = 0) -> dict:
    """8.E — SGP loss count zero in Phase 1; equals train rallies in Phase 2."""
    assert phase1_sgp_count == 0, (
        f"VIOLATION (8.E): Phase 1 SGP loss count {phase1_sgp_count} != 0")
    assert phase2_sgp_count == phase2_train_rally_count, (
        f"VIOLATION (8.E): Phase 2 SGP loss count {phase2_sgp_count} != "
        f"train rally count {phase2_train_rally_count}")
    assert phase2_test_sgp_count == 0, (
        f"VIOLATION (8.E): Phase 2 applied SGP to {phase2_test_sgp_count} test rallies")
    return {
        "audit_8E_sgp_loss_count": "PASS",
        "phase1_sgp_loss_count": phase1_sgp_count,
        "phase2_sgp_loss_count": phase2_sgp_count,
        "phase2_train_rally_count": phase2_train_rally_count,
    }


def audit_train_val_match_disjoint(train_match_set: set,
                                    val_match_set: set) -> dict:
    """Sanity — Fold-1 train and val match groups must be disjoint."""
    overlap = train_match_set & val_match_set
    assert not overlap, (
        f"VIOLATION (Fold split): train and val match groups overlap "
        f"({len(overlap)} matches)")
    return {
        "audit_train_val_match_disjoint": "PASS",
        "n_train_matches": len(train_match_set),
        "n_val_matches": len(val_match_set),
    }
