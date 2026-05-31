"""Blend analysis for `_oldtest` variants (2026-05-13).

After the AICUP organizers' 2026-05-13 announcement permitting
`data/test.csv` (OLD test) as additional training data, we
re-trained:
  - v11_mulminet_aug -> v11_mulminet_aug_oldtest
  - v14_seed2        -> v14_seed2_oldtest
  - v16_testhist_aug -> v16_testhist_aug_oldtest

This script compares the LB-best 5-component NONE blend
`(v11_aug, v11plus, v13, v14_seed2, v16_avg3)` against
single-component swaps where one slot is replaced with its
`_oldtest` counterpart, plus the full 3-way swap.

Outputs a ranking and (if `--write-submission`) the top single-swap
candidate as a NONE-calibration CSV.

USAGE:
  python -u src/analyze_oldtest_blend.py
  python -u src/analyze_oldtest_blend.py --write-submission
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, SUBMISSION_DIR  # noqa

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")

# Match blend_zoo_v2 conventions
N_ACTION = 19
N_POINT = 10
ACTION_EVAL = list(range(15))
POINT_EVAL = list(range(10))

# LB-best subset (zoo_v10 elig2, NONE, 0.3694391)
LB_BEST_SUBSET = ["v11_aug", "v11plus", "v13", "v14_seed2", "v16_avg3"]
LB_BEST_OOF = 0.3771
LB_BEST_LB = 0.3694391
LB_BEST_RATIO = LB_BEST_LB / LB_BEST_OOF  # ~0.9810

# Mapping of LB-best component -> proposed _oldtest replacement.
# v11_aug -> v11_aug_oldtest is a like-for-like swap (same arch, +oldtest).
# v11plus -> v11_mulminet_aug_oldtest is structural (different arch+aug, +oldtest).
# v13 -> v13_oldtest is a like-for-like GBM swap (same arch, +oldtest).
# v14_seed2 -> v14_seed2_oldtest is like-for-like.
# v16_avg3 -> v16_testhist_aug_oldtest is structural (avg-of-3-seeds -> single seed
#   with test-history aug + oldtest).
SWAPS = {
    "v11_aug":   "v11_aug_oldtest",            # like-for-like transformer
    "v11plus":   "v11_mulminet_aug_oldtest",    # structural diversity slot
    "v13":       "v13_oldtest",                 # like-for-like GBM
    "v14_seed2": "v14_seed2_oldtest",           # like-for-like GBM
    "v16_avg3":  "v16_testhist_aug_oldtest",    # structural GBM-aug
}


def pad_act19(arr: np.ndarray) -> np.ndarray:
    if arr.shape[1] >= N_ACTION:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((arr.shape[0], N_ACTION), dtype=np.float32)
    out[:, : arr.shape[1]] = arr
    return out


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray,
                  labels: List[int], n_total: int) -> float:
    cm = np.bincount(y_true.astype(np.int64) * n_total + y_pred.astype(np.int64),
                     minlength=n_total * n_total).reshape(n_total, n_total)
    col_sum = cm.sum(axis=0)
    row_sum = cm.sum(axis=1)
    diag = np.diag(cm)
    f1s = np.zeros(len(labels), dtype=np.float64)
    for i, c in enumerate(labels):
        tp = diag[c]
        fp = col_sum[c] - tp
        fn = row_sum[c] - tp
        denom = 2 * tp + fp + fn
        f1s[i] = 0.0 if denom <= 0 else (2 * tp) / denom
    return float(f1s.mean())


def load_components(tags: List[str]):
    """Load OOF + test arrays for the requested tags. Reference is v11_aug,
    which is always present in our LB-best subset.

    NOTE: `_oldtest` tags have extended OOF arrays (72065 rows vs standard 69712)
    because old-test rows were added to training data. We slice the first
    `N_REF` rows to align with the standard 69712 OOF index. Verified that the
    first N_REF y_act/y_pt/y_srv labels are identical to v11_aug's.
    """
    ref = "v11_aug"
    y_a = np.load(f"{OOF_DIR}/{ref}_oof_y_act.npy")
    y_p = np.load(f"{OOF_DIR}/{ref}_oof_y_pt.npy")
    y_s = np.load(f"{OOF_DIR}/{ref}_oof_y_srv.npy")
    test_uid = np.load(f"{OOF_DIR}/{ref}_test_rally_uid.npy")
    mask = np.load(f"{OOF_DIR}/{ref}_oof_mask.npy")
    N_REF = len(y_a)

    comp: Dict[str, Dict[str, np.ndarray]] = {}
    for tag in tags:
        path_act = f"{OOF_DIR}/{tag}_oof_act.npy"
        if not os.path.exists(path_act):
            print(f"  [missing] {tag} OOF not found at {path_act}")
            continue
        oof_act = np.load(path_act)
        oof_pt = np.load(f"{OOF_DIR}/{tag}_oof_pt.npy")
        oof_srv = np.load(f"{OOF_DIR}/{tag}_oof_srv.npy")

        # If this is a `_oldtest` tag, slice first N_REF rows to align indices.
        if oof_act.shape[0] != N_REF:
            if oof_act.shape[0] < N_REF:
                raise AssertionError(
                    f"{tag} OOF has fewer rows ({oof_act.shape[0]}) than ref ({N_REF}); "
                    f"cannot align")
            # Verify y labels match for sliced region
            tag_y_path = f"{OOF_DIR}/{tag}_oof_y_act.npy"
            if os.path.exists(tag_y_path):
                tag_y = np.load(tag_y_path)
                if not np.array_equal(tag_y[:N_REF], y_a):
                    raise AssertionError(
                        f"{tag} oldtest slice [:{N_REF}] y_act != reference; "
                        f"row order assumption broken")
            print(f"  [{tag}] aligning OOF: slicing {oof_act.shape[0]} -> {N_REF} "
                  f"(dropped last {oof_act.shape[0] - N_REF} old-test rows)")
            oof_act = oof_act[:N_REF]
            oof_pt = oof_pt[:N_REF]
            oof_srv = oof_srv[:N_REF]

        d: Dict[str, np.ndarray] = {}
        d["oof_act"] = pad_act19(oof_act)
        d["oof_pt"] = oof_pt.astype(np.float32, copy=False)
        d["oof_srv"] = oof_srv.astype(np.float32, copy=False)
        d["test_act"] = pad_act19(np.load(f"{OOF_DIR}/{tag}_test_act.npy"))
        d["test_pt"] = np.load(f"{OOF_DIR}/{tag}_test_pt.npy").astype(np.float32, copy=False)
        d["test_srv"] = np.load(f"{OOF_DIR}/{tag}_test_srv.npy").astype(np.float32, copy=False)

        # Sanity checks vs reference
        skip_tag = False
        tag_uid_path = f"{OOF_DIR}/{tag}_test_rally_uid.npy"
        if os.path.exists(tag_uid_path):
            tag_uid = np.load(tag_uid_path)
            if not np.array_equal(tag_uid, test_uid):
                if os.environ.get("ALLOW_UID_MISMATCH", "0") == "1":
                    print(f"  [warn] {tag} test rally_uid mismatch — SKIPPING this tag (ALLOW_UID_MISMATCH=1)")
                    skip_tag = True
                else:
                    raise AssertionError(f"Test rally_uid mismatch for tag {tag}")
        if skip_tag:
            continue

        bad_field = None
        for nm, arr in d.items():
            if not np.isfinite(arr).all():
                bad_field = nm
                break
        if bad_field is not None:
            if os.environ.get("ALLOW_NONFINITE", "0") == "1":
                print(f"  [warn] non-finite in {tag}_{bad_field} — SKIPPING this tag")
                continue
            raise AssertionError(f"Non-finite values in {tag}_{bad_field}")

        comp[tag] = d

    return comp, y_a, y_p, y_s, mask, test_uid


def evaluate_subset_none(subset: List[str], comp: Dict, y_a, y_p, y_s,
                         optimize: bool = True, n_samples: int = 200,
                         seed: int = 20260513):
    """Per-task Dirichlet-search NONE blend. Mirrors blend_zoo_v2 NONE
    calibration (no temperature, no per-class weights, but per-task weights
    are searched independently).

    optimize=False → equal-weight blend (lightweight comparison).
    optimize=True  → per-task Dirichlet random search (slower, matches
                     blend_zoo_v2 LB-best 0.3771 OOF for the LB-best subset).
    """
    n = len(subset)
    rng = np.random.default_rng(seed)

    act_stack = np.stack([comp[t]["oof_act"] for t in subset], axis=0)
    pt_stack = np.stack([comp[t]["oof_pt"] for t in subset], axis=0)
    srv_stack = np.stack([comp[t]["oof_srv"] for t in subset], axis=0)

    if not optimize:
        w = np.full(n, 1.0 / n)
        blend_a = (w[:, None, None] * act_stack).sum(axis=0)
        blend_p = (w[:, None, None] * pt_stack).sum(axis=0)
        blend_s = (w[:, None] * srv_stack).sum(axis=0)
        f_a = fast_macro_f1(y_a, blend_a.argmax(axis=1), ACTION_EVAL, N_ACTION)
        f_p = fast_macro_f1(y_p, blend_p.argmax(axis=1), POINT_EVAL, N_POINT)
        auc = roc_auc_score(y_s, blend_s)
        ov = 0.4 * f_a + 0.4 * f_p + 0.2 * auc
        return {"F1_a": f_a, "F1_p": f_p, "AUC": auc, "OV": ov,
                "w_a": w, "w_p": w, "w_s": w}

    # Per-task Dirichlet random search (independent weights per task)
    best_a, best_w_a = -1.0, np.full(n, 1.0 / n)
    best_p, best_w_p = -1.0, np.full(n, 1.0 / n)
    best_s, best_w_s = -1.0, np.full(n, 1.0 / n)

    # Always evaluate equal weight first
    blend_a0 = (best_w_a[:, None, None] * act_stack).sum(axis=0)
    blend_p0 = (best_w_p[:, None, None] * pt_stack).sum(axis=0)
    blend_s0 = (best_w_s[:, None] * srv_stack).sum(axis=0)
    best_a = fast_macro_f1(y_a, blend_a0.argmax(axis=1), ACTION_EVAL, N_ACTION)
    best_p = fast_macro_f1(y_p, blend_p0.argmax(axis=1), POINT_EVAL, N_POINT)
    best_s = roc_auc_score(y_s, blend_s0)

    for _ in range(n_samples):
        w = rng.dirichlet(np.ones(n))
        ba = (w[:, None, None] * act_stack).sum(axis=0)
        f = fast_macro_f1(y_a, ba.argmax(axis=1), ACTION_EVAL, N_ACTION)
        if f > best_a:
            best_a, best_w_a = f, w
        bp = (w[:, None, None] * pt_stack).sum(axis=0)
        f = fast_macro_f1(y_p, bp.argmax(axis=1), POINT_EVAL, N_POINT)
        if f > best_p:
            best_p, best_w_p = f, w
        bs = (w[:, None] * srv_stack).sum(axis=0)
        a = roc_auc_score(y_s, bs)
        if a > best_s:
            best_s, best_w_s = a, w

    ov = 0.4 * best_a + 0.4 * best_p + 0.2 * best_s
    return {"F1_a": best_a, "F1_p": best_p, "AUC": best_s, "OV": ov,
            "w_a": best_w_a, "w_p": best_w_p, "w_s": best_w_s}


def build_none_test(subset: List[str], comp: Dict, w_a=None, w_p=None, w_s=None):
    """Mirror blend_zoo_v2's NONE submission build with optional per-task weights."""
    n = len(subset)
    eq = np.full(n, 1.0 / n)
    if w_a is None: w_a = eq
    if w_p is None: w_p = eq
    if w_s is None: w_s = eq
    act_stack = np.stack([comp[t]["test_act"] for t in subset], axis=0)
    pt_stack = np.stack([comp[t]["test_pt"] for t in subset], axis=0)
    srv_stack = np.stack([comp[t]["test_srv"] for t in subset], axis=0)
    blend_a = (w_a[:, None, None] * act_stack).sum(axis=0)
    blend_p = (w_p[:, None, None] * pt_stack).sum(axis=0)
    blend_s = (w_s[:, None] * srv_stack).sum(axis=0)
    return blend_a.argmax(axis=1), blend_p.argmax(axis=1), blend_s


def write_submission(test_uid, pred_a, pred_p, blend_s, fname: str):
    out = pd.DataFrame({
        "rally_uid": test_uid,
        "actionId": pred_a.astype(int),
        "pointId": pred_p.astype(int),
        "serverGetPoint": np.clip(blend_s, 0.0, 1.0).astype(np.float32),
    })
    path = os.path.join(SUBMISSION_DIR, fname)
    out.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"  Wrote {path} ({len(out)} rows)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-submission", action="store_true",
                        help="If set, materialise the top single-swap NONE candidate as a CSV.")
    parser.add_argument("--write-aggressive", action="store_true",
                        help="If set, also materialise the all-three-swap NONE candidate.")
    parser.add_argument("--no-optimize", action="store_true",
                        help="Use equal-weight NONE blend (fast, ~3s). Default: per-task "
                             "Dirichlet search (slower, matches blend_zoo_v2 LB-best 0.3771).")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Dirichlet samples per task (default: 200).")
    args = parser.parse_args()
    optimize = not args.no_optimize

    print("=" * 70)
    print(" Old-test variant blend analysis (2026-05-13)")
    print("=" * 70)
    print(f" LB-best subset (NONE): {LB_BEST_SUBSET}")
    print(f" LB-best OOF: {LB_BEST_OOF:.4f}  LB: {LB_BEST_LB:.7f}  ratio: {LB_BEST_RATIO:.4f}")
    print()

    # Load all required tags
    all_tags = list(LB_BEST_SUBSET) + list(SWAPS.values())
    comp, y_a, y_p, y_s, mask, test_uid = load_components(all_tags)

    available_swaps = {k: v for k, v in SWAPS.items() if v in comp}
    missing = {k: v for k, v in SWAPS.items() if v not in comp}

    print(f" Available swaps: {len(available_swaps)}/{len(SWAPS)}")
    for k, v in available_swaps.items():
        print(f"   {k} -> {v}")
    if missing:
        print(f" Missing (not yet trained):")
        for k, v in missing.items():
            print(f"   {k} -> {v}  [no OOF found]")
    print()

    # Baseline (LB-best subset, NONE)
    print(" --- Baseline (LB-best NONE) ---")
    base = evaluate_subset_none(LB_BEST_SUBSET, comp, y_a, y_p, y_s,
                                 optimize=optimize, n_samples=args.n_samples)
    print(f"   subset: {LB_BEST_SUBSET}")
    print(f"   F1_a={base['F1_a']:.4f}  F1_p={base['F1_p']:.4f}  AUC={base['AUC']:.4f}  OV={base['OV']:.4f}")
    print()

    rows: List[Dict] = []
    rows.append({
        "label": "BASELINE (LB-best)",
        "subset": LB_BEST_SUBSET,
        **base,
        "delta_OV": 0.0,
        "predicted_LB": base["OV"] * LB_BEST_RATIO,
    })

    # Single swaps
    for orig, new in available_swaps.items():
        new_subset = [new if t == orig else t for t in LB_BEST_SUBSET]
        m = evaluate_subset_none(new_subset, comp, y_a, y_p, y_s,
                                  optimize=optimize, n_samples=args.n_samples)
        delta = m["OV"] - base["OV"]
        rows.append({
            "label": f"SWAP {orig}->{new}",
            "subset": new_subset,
            **m,
            "delta_OV": delta,
            "predicted_LB": m["OV"] * LB_BEST_RATIO,
        })

    # All-available swap (if at least 2 available)
    if len(available_swaps) >= 2:
        new_subset = [available_swaps.get(t, t) for t in LB_BEST_SUBSET]
        m = evaluate_subset_none(new_subset, comp, y_a, y_p, y_s,
                                  optimize=optimize, n_samples=args.n_samples)
        delta = m["OV"] - base["OV"]
        rows.append({
            "label": f"SWAP all-{len(available_swaps)}-available",
            "subset": new_subset,
            **m,
            "delta_OV": delta,
            "predicted_LB": m["OV"] * LB_BEST_RATIO,
        })

    # Pairs (most informative if single swaps don't help but a pair does)
    if len(available_swaps) >= 2:
        keys = list(available_swaps.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                pair = {keys[i]: available_swaps[keys[i]],
                        keys[j]: available_swaps[keys[j]]}
                new_subset = [pair.get(t, t) for t in LB_BEST_SUBSET]
                m = evaluate_subset_none(new_subset, comp, y_a, y_p, y_s,
                                  optimize=optimize, n_samples=args.n_samples)
                delta = m["OV"] - base["OV"]
                rows.append({
                    "label": f"PAIR {keys[i]}+{keys[j]}",
                    "subset": new_subset,
                    **m,
                    "delta_OV": delta,
                    "predicted_LB": m["OV"] * LB_BEST_RATIO,
                })

    # Print ranking
    mode = f"Dirichlet({args.n_samples}) per-task" if optimize else "equal-weight"
    print(f" --- Ranking by OOF OV ({mode} NONE blend) ---")
    rows_sorted = sorted(rows, key=lambda r: -r["OV"])
    print(f" {'#':>2}  {'label':<40}  {'F1_a':>6}  {'F1_p':>6}  {'AUC':>6}  {'OV':>6}  {'dOV':>7}  {'pred_LB':>7}")
    print(" " + "-" * 95)
    for i, r in enumerate(rows_sorted, start=1):
        print(f" {i:>2}  {r['label']:<40}  {r['F1_a']:.4f}  {r['F1_p']:.4f}  {r['AUC']:.4f}  "
              f"{r['OV']:.4f}  {r['delta_OV']:+.4f}  {r['predicted_LB']:.4f}")
    print()
    if optimize:
        print(" Optimized NONE blend: per-task Dirichlet weight search (matches blend_zoo_v2).")
        print(f" Baseline OOF should be close to 0.3771 (zoo_v10 elig2 LB-best). Got: {base['OV']:.4f}.")
    else:
        print(" Equal-weight NONE blend = simplified comparison. Re-run without --no-optimize")
        print(" for full Dirichlet search.")
    print()

    # Save ranking CSV
    rank_df = pd.DataFrame([{**r, "subset": ",".join(r["subset"])} for r in rows_sorted])
    rank_path = os.path.join(SUBMISSION_DIR, "oldtest_blend_ranking.csv")
    rank_df.to_csv(rank_path, index=False)
    print(f" Ranking saved -> {rank_path}")
    print()

    # Optionally materialise submissions
    if args.write_submission or args.write_aggressive:
        # Recompute available_swaps in test space
        single_swap_rows = [r for r in rows_sorted
                            if r["label"].startswith("SWAP ") and "all-3" not in r["label"]
                            and r["delta_OV"] > 0]
        if not single_swap_rows:
            print(" No single-swap candidate showed OOF gain. NOT writing single-swap submission.")
        else:
            top = single_swap_rows[0]
            tag_short = "_".join([t.replace("_oldtest", "").replace("_", "")
                                  if t.endswith("_oldtest") else t.replace("_", "")
                                  for t in top["subset"]])
            fname = f"submission_OLDTEST_SAFE_NONE_{tag_short}.csv"
            print(f" Building single-swap candidate: {top['label']}")
            pa, pp, ps = build_none_test(top["subset"], comp,
                                          w_a=top.get("w_a"), w_p=top.get("w_p"), w_s=top.get("w_s"))
            write_submission(test_uid, pa, pp, ps, fname)

        if args.write_aggressive:
            agg_rows = [r for r in rows_sorted if r["label"] == "SWAP all-3"]
            if agg_rows:
                top = agg_rows[0]
                fname = "submission_OLDTEST_AGGRESSIVE_NONE_all3.csv"
                print(f" Building aggressive all-3 candidate: {top['label']}")
                pa, pp, ps = build_none_test(top["subset"], comp,
                                              w_a=top.get("w_a"), w_p=top.get("w_p"), w_s=top.get("w_s"))
                write_submission(test_uid, pa, pp, ps, fname)

    print()
    print(" === Reading guide ===")
    print(f" - LB-best baseline OOF: {base['OV']:.4f}, LB: {LB_BEST_LB:.7f}")
    print(" - Single-swap candidates with delta_OV > +0.002 are LB candidates per Workflow §4.6.")
    print(" - Single-swap candidates with delta_OV > 0 but < +0.002 are diagnostic-only.")
    print(" - All-3 swap is HIGH-RISK (3 simultaneous structural changes); diagnostic-only unless")
    print("   delta_OV >> +0.005.")
    print(" - The pattern from R-007/R-008/R-016/R-017/R-020b/R-026 (6 instances) is that blender-")
    print("   search OOF gains via re-arrangement DO NOT TRANSFER to LB. Old-test retraining is a")
    print("   STRUCTURAL change to the underlying model, not a re-arrangement, so this exception")
    print("   may apply — but verify with single-component swap LB upload before all-3 swap.")
    print(" - REQUIRED: Codex review (R-### entry) BEFORE any LB upload.")


if __name__ == "__main__":
    main()
