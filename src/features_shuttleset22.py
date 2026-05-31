"""features_shuttleset22 — loader for badminton ShuttleSet22 dataset (R-021).

External-data pretraining source per R-021 / Codex APPROVE_WITH_FIXES (2026-05-12).

Loads ShuttleSet22 (Wang et al., KDD 2023, arXiv 2306.15664) for use as
PRETRAINING ONLY of v11_mulminet's transformer encoder. Per Codex P1.3:
- ENCODER weights transfer to AI CUP fine-tune
- Label heads (badminton vocabulary) DO NOT transfer
- Player metadata DO NOT transfer (different player pool)

Source data: data/external/CoachAI-Projects/CoachAI-Challenge-IJCAI2023/ShuttleSet22/
Schema (per ShuttleSet22 README.md):
  - 60 match folders, each with set{1,2,3}.csv
  - Per-shot row: rally, ball_round, time, player(A/B), server, type (Chinese badminton),
    aroundhead, backhand, hit_area, landing_area, landing_x/y, getpoint_player, ...

We extract per-rally sequences of:
  - badminton_type (categorical, ~18 classes)
  - landing_area (categorical 1..9 standard, 10 = outside)
  - hit_area (categorical 1..7 + outside)
  - backhand (0/1)
  - aroundhead (0/1)
  - server (0/1)

NOT used:
  - landing_x/y (continuous coords — encoder only needs categorical)
  - player_location_x/y (player-specific, not transferable)
  - player A/B identity (would create badminton-player embeddings — not transferable)
  - getpoint_player (rally-level outcome; could be aux task but not required for encoder pretrain)

Total expected: ~33,612 strokes / 3,992 rallies / 35 players (per paper).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allowed source columns from ShuttleSet22 (per Codex P3.5-style audit list).
SOURCE_COLS = [
    "rally", "ball_round", "player", "server",
    "type", "aroundhead", "backhand",
    "hit_area", "landing_area",
]
# Forbidden columns we MUST NOT read (per Codex P1.3):
FORBIDDEN_SOURCE = {
    "landing_x", "landing_y",        # raw coords — we use categorical
    "player_location_x", "player_location_y",
    "opponent_location_x", "opponent_location_y",
    "getpoint_player",               # rally outcome — not needed for encoder pretrain
    "frame_num", "time",             # timing meta
}

# Badminton stroke type vocabulary from ShuttleSet22 README.
# Total: 18 + 1 unknown placeholder = 19 classes (we'll map to dense int).
BADMINTON_TYPES = [
    "發短球",      # short service
    "發長球",      # long service
    "放小球",      # net shot
    "擋小球",      # return net
    "殺球",        # smash
    "點扣",        # wrist smash
    "挑球",        # lob
    "防守回挑",    # defensive return lob
    "長球",        # clear
    "平球",        # drive
    "小平球",      # driven flight
    "後場抽平球",  # back-court drive
    "切球",        # drop
    "過渡切球",    # passive drop
    "推球",        # push
    "撲球",        # rush
    "防守回抽",    # defensive return drive
    "勾球",        # cross-court net shot
]
TYPE_TO_IDX = {t: i + 1 for i, t in enumerate(BADMINTON_TYPES)}  # 1-indexed; 0 = unknown


def load_shuttleset22(root: str | Path) -> pd.DataFrame:
    """Walk ShuttleSet22/set/ directory, concatenate all matches/sets.

    Returns a single DataFrame with columns from SOURCE_COLS plus:
      - match_id (int): unique per match folder
      - set_num (int): 1, 2, or 3
      - rally_uid (str): "{match_id}_{set}_{rally}"
      - type_idx (int): integer encoding via TYPE_TO_IDX (0 = unknown)

    Asserts no FORBIDDEN_SOURCE columns are returned in the output.
    """
    root = Path(root)
    set_dir = root / "set"
    if not set_dir.is_dir():
        raise FileNotFoundError(f"ShuttleSet22 set dir not found at {set_dir}")

    rows = []
    match_dirs = sorted([d for d in set_dir.iterdir() if d.is_dir()])
    print(f"  [shuttleset22] scanning {len(match_dirs)} match folders...")

    for match_id, match_dir in enumerate(match_dirs):
        for set_csv in sorted(match_dir.glob("set*.csv")):
            try:
                df = pd.read_csv(set_csv)
            except Exception as e:
                print(f"  [shuttleset22] WARN failed to read {set_csv}: {e}")
                continue
            # Restrict to allowed columns only
            cols = [c for c in SOURCE_COLS if c in df.columns]
            df = df[cols].copy()
            # Verify no forbidden columns (defensive)
            forbidden_present = set(df.columns) & FORBIDDEN_SOURCE
            assert not forbidden_present, \
                f"VIOLATION: {set_csv} returned forbidden columns: {forbidden_present}"
            df["match_id"] = match_id
            set_num_str = set_csv.stem.replace("set", "")
            try:
                df["set_num"] = int(set_num_str)
            except ValueError:
                df["set_num"] = 0
            rows.append(df)

    if not rows:
        raise RuntimeError("No ShuttleSet22 data loaded")

    out = pd.concat(rows, ignore_index=True)

    # Build rally_uid
    out["rally_uid"] = (out["match_id"].astype(str) + "_" +
                        out["set_num"].astype(str) + "_" +
                        out["rally"].astype(str))

    # Map type to integer (0 = unknown / missing)
    out["type_idx"] = out["type"].map(TYPE_TO_IDX).fillna(0).astype(int)

    # Final audit: assert no forbidden cols leaked
    forbidden_in_output = set(out.columns) & FORBIDDEN_SOURCE
    assert not forbidden_in_output, \
        f"VIOLATION: output contains forbidden cols: {forbidden_in_output}"

    return out


def build_rally_sequences(df: pd.DataFrame, max_len: int = 32,
                          min_rally_len: int = 2) -> list:
    """Group strokes into per-rally sequences. Returns list of dicts:
      - rally_uid: str
      - shots: ndarray (T, 6) int8 of [type_idx, landing_area, hit_area,
                                       backhand, aroundhead, server]
      - length: int (T)
    """
    seqs = []
    n_skipped = 0
    for rally_uid, grp in df.groupby("rally_uid", sort=False):
        grp = grp.sort_values("ball_round").reset_index(drop=True)
        T = len(grp)
        if T < min_rally_len:
            n_skipped += 1
            continue
        T = min(T, max_len)
        type_idx = grp["type_idx"].to_numpy()[:T].astype(np.int8)
        # Categorical: clip outside / NaN
        landing = grp.get("landing_area", pd.Series([0]*T))
        landing = landing.fillna(0).clip(lower=0, upper=10).astype(int).to_numpy()[:T].astype(np.int8)
        hit = grp.get("hit_area", pd.Series([0]*T))
        hit = hit.fillna(0).clip(lower=0, upper=10).astype(int).to_numpy()[:T].astype(np.int8)
        backhand = grp.get("backhand", pd.Series([0]*T))
        backhand = backhand.fillna(0).astype(int).to_numpy()[:T].astype(np.int8)
        aroundhead = grp.get("aroundhead", pd.Series([0]*T))
        aroundhead = aroundhead.fillna(0).astype(int).to_numpy()[:T].astype(np.int8)
        server = grp.get("server", pd.Series([0]*T))
        server = server.fillna(0).astype(int).to_numpy()[:T].astype(np.int8)
        shots = np.stack([type_idx, landing, hit, backhand, aroundhead, server],
                         axis=1).astype(np.int8)
        seqs.append({
            "rally_uid": rally_uid,
            "shots": shots,
            "length": int(T),
        })
    print(f"  [shuttleset22] built {len(seqs)} rally sequences "
          f"(skipped {n_skipped} short rallies)")
    return seqs


def split_train_val(seqs: list, val_frac: float = 0.1, seed: int = 51966) -> tuple:
    """Random train/val split at rally level."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(seqs))
    n_val = max(1, int(len(seqs) * val_frac))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return [seqs[i] for i in tr_idx], [seqs[i] for i in val_idx]


def audit_dataset(df: pd.DataFrame, seqs: list) -> dict:
    """Build a dictionary of audit results for runs/r021_audit.json."""
    audit = {
        "n_strokes_raw": int(len(df)),
        "n_matches": int(df["match_id"].nunique()),
        "n_rallies_raw": int(df["rally_uid"].nunique()),
        "n_rallies_kept": int(len(seqs)),
        "stroke_type_distribution": df["type"].value_counts().head(20).to_dict(),
        "type_idx_unknown_pct": float((df["type_idx"] == 0).mean() * 100),
        "shot_count_quantiles": {
            "p25": float(np.quantile([s["length"] for s in seqs], 0.25)),
            "p50": float(np.quantile([s["length"] for s in seqs], 0.50)),
            "p75": float(np.quantile([s["length"] for s in seqs], 0.75)),
            "p95": float(np.quantile([s["length"] for s in seqs], 0.95)),
            "max": int(max(s["length"] for s in seqs)),
        },
        "source_cols_used": SOURCE_COLS,
        "forbidden_cols": sorted(FORBIDDEN_SOURCE),
    }
    return audit


if __name__ == "__main__":
    # Quick CLI: dump audit to runs/r021_audit.json
    import json
    root = "data/external/CoachAI-Projects/CoachAI-Challenge-IJCAI2023/ShuttleSet22"
    print("Loading ShuttleSet22 from:", root)
    df = load_shuttleset22(root)
    print(f"Loaded {len(df)} stroke rows from {df['match_id'].nunique()} matches")
    seqs = build_rally_sequences(df)
    audit = audit_dataset(df, seqs)
    out_dir = Path("runs/r021_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "shuttleset22_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, default=str, ensure_ascii=False),
                          encoding="utf-8")
    print(f"Audit saved to {audit_path}")
    print(json.dumps(audit, indent=2, default=str, ensure_ascii=False))
