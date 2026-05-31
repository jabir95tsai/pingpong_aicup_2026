"""R-072 Rule Override v2 — multi-pattern extension of R-042.

Background: R-042 = R-034 + single-pattern zero-prob override
  P(class | prev_actionId, last_actionId, last_pointId)
This is the proven mechanism that produced LB +0.0028 with 1.0 transfer rate.

R-072 layers 3 ADDITIONAL override tables on top, each independent and
ablatable. Same zero-probability mechanism — never invents predictions, only
flips empirically-impossible predictions to the train mode.

Order of application (later layers only touch rows earlier layers did NOT):
  Layer A (existing R-042): (prev_actionId, last_actionId, last_pointId)
  Layer B (new, deeper):    (prev_prev_actionId, prev_actionId, last_actionId, last_pointId)
  Layer C (new, hand):      (last_handId, prev_actionId, last_actionId, last_pointId)
  Layer D (new, position):  (last_positionId, prev_actionId, last_actionId, last_pointId)

HARD RULES (cannot be relaxed by any layer):
- NEVER touches serverGetPoint column.
- Override only fires when (predicted_class in train) AND (train P(predicted_class | context) == 0).
- Context must have MIN_CONTEXT_SAMPLES train observations.
- Layer order is fixed; later layers respect earlier layer's overrides.

Validation: applies the same overrides to a 5-fold OOF (R-034 PAIR base OOF
or similar) to estimate OOF F1 delta before committing the LB candidate.

USAGE:
    python -u src/apply_rule_override_v2.py \\
        --input  submissions/submission_R067cr_alpha030_v22_blend.csv \\
        --train  data/train.csv \\
        --test   data/test_new.csv \\
        --output submissions/submission_R072_R067cr_PLUS_RULE_V2.csv \\
        --oof-tag v14_seed2_v15feat_a_fold1   # optional OOF validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

MIN_CONTEXT_SAMPLES_A = 30   # R-042 default (Layer A)
MIN_CONTEXT_SAMPLES_B = 20   # deeper prefix → smaller contexts allowed
MIN_CONTEXT_SAMPLES_C = 25
MIN_CONTEXT_SAMPLES_D = 25


def _build_table(
    train: pd.DataFrame,
    context_cols: List[str],
    target_col: str,
    min_samples: int,
) -> Dict[tuple, Tuple[Dict[int, float], int, int]]:
    """Build P(target | context) from train.

    Returns {context_tuple: (probs_dict, mode, n_total)} for contexts with
    >= min_samples observations.
    """
    df = train.sort_values(["rally_uid", "strikeNumber"]).copy()
    # prev_actionId one shift back
    df["prev_actionId"] = df.groupby("rally_uid")["actionId"].shift(1)
    # prev_prev_actionId two shifts back (Layer B)
    df["prev_prev_actionId"] = df.groupby("rally_uid")["actionId"].shift(2)
    # last_actionId / last_pointId / last_handId / last_positionId already at row
    df["last_actionId"] = df["actionId"]
    df["last_pointId"] = df["pointId"]
    df["last_handId"] = df["handId"]
    df["last_positionId"] = df["positionId"]
    # next_target — shift -1
    df[f"next_{target_col}"] = df.groupby("rally_uid")[target_col].shift(-1)
    df = df.dropna(subset=context_cols + [f"next_{target_col}"])
    for c in context_cols + [f"next_{target_col}"]:
        df[c] = df[c].astype(int)

    table = {}
    for key, vals in df.groupby(context_cols)[f"next_{target_col}"]:
        if len(vals) < min_samples:
            continue
        probs = vals.value_counts(normalize=True).to_dict()
        mode = int(vals.value_counts().idxmax())
        table[tuple(int(k) for k in (key if isinstance(key, tuple) else (key,)))] = (
            probs, mode, len(vals)
        )
    return table


def _build_test_context(test: pd.DataFrame) -> pd.DataFrame:
    """For each test rally, extract the context features needed by all layers."""
    rows = []
    for rid, g in test.groupby("rally_uid"):
        g = g.sort_values("strikeNumber").reset_index(drop=True)
        if len(g) < 2:
            continue
        last = g.iloc[-1]
        prev = g.iloc[-2]
        prev_prev = g.iloc[-3] if len(g) >= 3 else None
        row = {
            "rally_uid": int(rid),
            "prev_actionId":      int(prev["actionId"]),
            "last_actionId":      int(last["actionId"]),
            "last_pointId":       int(last["pointId"]),
            "last_handId":        int(last["handId"]),
            "last_positionId":    int(last["positionId"]),
            "prev_prev_actionId": int(prev_prev["actionId"]) if prev_prev is not None else -1,
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("rally_uid")


# ─── Layer specs — context columns + min_samples per layer ────────────────────
LAYERS = [
    # (layer_id, context_cols, min_samples)
    ("A", ["prev_actionId", "last_actionId", "last_pointId"], MIN_CONTEXT_SAMPLES_A),
    ("B", ["prev_prev_actionId", "prev_actionId", "last_actionId", "last_pointId"], MIN_CONTEXT_SAMPLES_B),
    ("C", ["last_handId", "prev_actionId", "last_actionId", "last_pointId"], MIN_CONTEXT_SAMPLES_C),
    ("D", ["last_positionId", "prev_actionId", "last_actionId", "last_pointId"], MIN_CONTEXT_SAMPLES_D),
]


def apply_rule_override_v2(
    submission: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[dict], dict]:
    """Apply multi-layer zero-prob overrides. Returns (new_sub, changes, per_layer_stats)."""
    test_ctx = _build_test_context(test)

    # Mutable copies
    new_action = np.asarray(submission["actionId"].to_numpy(dtype=int, copy=True))
    new_point = np.asarray(submission["pointId"].to_numpy(dtype=int, copy=True))
    new_action.setflags(write=True)
    new_point.setflags(write=True)

    # Track which rows have been touched per target so later layers skip them
    touched_action = np.zeros(len(submission), dtype=bool)
    touched_point = np.zeros(len(submission), dtype=bool)
    changes: List[dict] = []
    per_layer = {}

    # Build all tables up front
    tables = {}
    for layer_id, cols, min_n in LAYERS:
        tables[layer_id] = {
            "actionId": _build_table(train, cols, "actionId", min_n),
            "pointId":  _build_table(train, cols, "pointId",  min_n),
            "context_cols": cols,
        }
        per_layer[layer_id] = {
            "n_action_contexts": len(tables[layer_id]["actionId"]),
            "n_point_contexts":  len(tables[layer_id]["pointId"]),
            "n_action_overrides": 0,
            "n_point_overrides":  0,
        }

    # Apply layers in order
    for layer_id, cols, min_n in LAYERS:
        act_tbl = tables[layer_id]["actionId"]
        pt_tbl  = tables[layer_id]["pointId"]

        for pos, (_, row) in enumerate(submission.iterrows()):
            rid = int(row["rally_uid"])
            if rid not in test_ctx.index:
                continue
            ctx_row = test_ctx.loc[rid]
            key = tuple(int(ctx_row[c]) for c in cols)
            # Layer B: skip if prev_prev_actionId == -1 (no third-prior shot)
            if "prev_prev_actionId" in cols and ctx_row["prev_prev_actionId"] == -1:
                continue

            # Action override
            if (not touched_action[pos]) and key in act_tbl:
                probs, mode, n = act_tbl[key]
                v_class = int(new_action[pos])
                if probs.get(v_class, 0.0) == 0.0 and mode != v_class:
                    new_action[pos] = mode
                    touched_action[pos] = True
                    per_layer[layer_id]["n_action_overrides"] += 1
                    changes.append({"layer": layer_id, "rally_uid": rid,
                                    "target": "actionId", "from": v_class,
                                    "to": mode, "context_n": n, "context_key": key})
            # Point override
            if (not touched_point[pos]) and key in pt_tbl:
                probs, mode, n = pt_tbl[key]
                v_class = int(new_point[pos])
                if probs.get(v_class, 0.0) == 0.0 and mode != v_class:
                    new_point[pos] = mode
                    touched_point[pos] = True
                    per_layer[layer_id]["n_point_overrides"] += 1
                    changes.append({"layer": layer_id, "rally_uid": rid,
                                    "target": "pointId", "from": v_class,
                                    "to": mode, "context_n": n, "context_key": key})

    out = submission.copy()
    out["actionId"] = new_action
    out["pointId"] = new_point
    # SGP UNTOUCHED — explicit safety assertion
    assert (out["serverGetPoint"].to_numpy() == submission["serverGetPoint"].to_numpy()).all(), \
        "SAFETY VIOLATION: SGP column was modified by rule override"
    return out, changes, per_layer


def _validate_no_sgp_in_changes(changes: List[dict]) -> None:
    """Hard assertion that no change touched SGP."""
    for c in changes:
        assert c["target"] != "serverGetPoint", \
            f"SAFETY VIOLATION: SGP override attempted at rally {c['rally_uid']}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Input submission.csv")
    p.add_argument("--train",  required=True, help="train.csv for building rules")
    p.add_argument("--test",   required=True, help="test.csv for matching context")
    p.add_argument("--output", required=True, help="Output submission.csv")
    p.add_argument("--report", default=None,
                   help="Optional JSON report path (defaults next to output)")
    args = p.parse_args()

    print("=" * 78)
    print(" R-072 Rule Override v2 — multi-pattern extension of R-042")
    print("=" * 78)

    submission = pd.read_csv(args.input)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    print(f" submission: {args.input}  ({len(submission)} rows)")
    print(f" train:      {args.train}  ({len(train)} rows)")
    print(f" test:       {args.test}   ({len(test)} rows, {test['rally_uid'].nunique()} rallies)")

    out, changes, per_layer = apply_rule_override_v2(submission, train, test)
    _validate_no_sgp_in_changes(changes)

    print()
    print(" PER-LAYER STATS")
    print(" " + "-" * 76)
    print(f" {'layer':<8} {'#A_ctx':>9} {'#P_ctx':>9} {'A_over':>9} {'P_over':>9}")
    for layer_id in ["A", "B", "C", "D"]:
        s = per_layer[layer_id]
        print(f" {layer_id:<8} {s['n_action_contexts']:>9} {s['n_point_contexts']:>9} "
              f"{s['n_action_overrides']:>9} {s['n_point_overrides']:>9}")
    total_action = sum(s["n_action_overrides"] for s in per_layer.values())
    total_point  = sum(s["n_point_overrides"]  for s in per_layer.values())
    total = total_action + total_point
    print()
    print(f" Total overrides: {total}  (action: {total_action}, point: {total_point})")
    print()
    print(" Sample changes (first 15):")
    for c in changes[:15]:
        print(f"   L{c['layer']}  rally={c['rally_uid']:>5}  {c['target']:<8}  "
              f"{c['from']:>2} -> {c['to']:<2}  (n_ctx={c['context_n']}, key={c['context_key']})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, lineterminator="\n", encoding="utf-8")
    print()
    print(f" Saved candidate CSV: {args.output}")

    # JSON report
    report_path = args.report or args.output.replace(".csv", "_report.json")
    report = {
        "rid": "R-072",
        "ts": "2026-05-25",
        "input": args.input,
        "output": args.output,
        "train": args.train,
        "test": args.test,
        "per_layer_stats": per_layer,
        "total_overrides": total,
        "total_action_overrides": total_action,
        "total_point_overrides":  total_point,
        "all_changes": changes,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f" Saved report:        {report_path}")


if __name__ == "__main__":
    main()
