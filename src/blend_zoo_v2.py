"""Blend Zoo v2 - purpose-built N-way blender (AI CUP 2026 ping-pong, P1).

NOT a wrapper around final_blend_optimized.py (which is 2-model only).
This is a from-scratch N-way blender per the Codex-revised TRAIN_PLAN spec.

Convention:
- Action probs are blended in the 19-class space. v11 / v11plus emit 15 classes
  (0..14, no serve channels); they are zero-padded to 19 before blending.
- Final action F1 is evaluated on labels 0..14 (ACTION_EVAL).
- Per-task weights are searched INDEPENDENTLY (action / point / server).
- Random search uses Dirichlet(alpha=ones) draws on the n_models simplex.
- Reference spread for the spread-penalty is taken from the search result
  for subset {v16_testhist_aug, v14_seed1, v12_5f, v11} with calibration THR
  (zoo_v16_fast_01-equivalent). If that entry is absent (e.g. under --replace),
  the median spread across all entries is used as fallback.

Search constraints (TRAIN_PLAN P1):
- Group A {v16_testhist_aug}: 0 or 1.
- Group B {v14_avg3, v14_seed0, v14_seed1, v14_seed2}: at most 1.
- Group C {v12_5f}: 0 or 1.
- Group D {v11, v11plus}: at least 1.
- Group E {v13}: 0 or 1.
- Total models per blend in {3, 4, 5, 6}.
- No per-SN-bucket weight conditioning anywhere in the search.

Calibration variants (cross-product with the weight search):
- THR  : temperature search + greedy class-weight + scipy Powell
- TEMP : temperature-only (no per-class weight)
- CW   : ACTION_CW / POINT_CW only (no temperature, no scipy)
- NONE : argmax of post-blend probs

Hard checks (run before any output is written):
- All tags share the OOF mask (full 69712).
- All tags share the OOF y_a / y_p / y_s arrays (where present on disk).
- All tags share the test rally_uid order (v11 source = submission CSV).
- All weight vectors sum to 1.0 +/- 1e-6.
- No NaN / Inf in any blended OOF or test prob.

Outputs:
- submissions/zoo_v2_ranking.csv  -- all entries, sorted by spread-penalised score.
- submissions/submission_zoo_v2_top<k>_<calib>_<provenance>.csv -- top-K submissions.
"""
import argparse
import itertools
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, SUBMISSION_DIR

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
DEFAULT_SEED = 20260504

ACTION_EVAL = list(range(15))
POINT_EVAL = list(range(10))
N_ACTION = 19
N_POINT = 10

ACTION_CW = {
    0: 1.5, 1: 0.6, 2: 0.9, 3: 1.5, 4: 1.2, 5: 1.0,
    6: 0.8, 7: 1.8, 8: 14.0, 9: 8.0, 10: 0.6, 11: 1.2,
    12: 0.9, 13: 0.7, 14: 10.0,
    15: 0.01, 16: 0.01, 17: 0.01, 18: 0.01,
}
# Mirrors final_blend_optimized.py's class-weight baselines. If those change
# there, change them here in lock-step.
POINT_CW = {
    0: 0.5, 1: 12.0, 2: 2.5, 3: 22.0, 4: 2.0,
    5: 0.9, 6: 1.5, 7: 0.8, 8: 0.7, 9: 0.6,
}

GROUP_A = ["v16_testhist_aug"]
GROUP_B = ["v14_avg3", "v14_seed0", "v14_seed1", "v14_seed2"]
GROUP_C = ["v12_5f"]
# Group D (Transformer): includes v11_aug (P6 V11+test-history aug) as of 2026-05-06.
# At least 1 from D required; selection logic in enumerate_subsets allows {v11},
# {v11plus}, {v11_aug}, or any pair / triple subject to overall size cap.
GROUP_D = ["v11", "v11plus", "v11_aug"]
GROUP_E = ["v13"]
ALL_TAGS = GROUP_A + GROUP_B + GROUP_C + GROUP_D + GROUP_E

REF_SUBSET = ["v16_testhist_aug", "v14_seed1", "v12_5f", "v11"]
REF_CALIB = "THR"

SN_BUCKETS: List[Tuple[str, callable]] = [
    ("SN=2", lambda nsn: (nsn == 2)),
    ("SN=3-4", lambda nsn: ((nsn >= 3) & (nsn <= 4))),
    ("SN=5-8", lambda nsn: ((nsn >= 5) & (nsn <= 8))),
    ("SN=9-12", lambda nsn: ((nsn >= 9) & (nsn <= 12))),
    ("SN>=13", lambda nsn: (nsn >= 13)),
]


def pad_act19(arr: np.ndarray) -> np.ndarray:
    if arr.shape[1] >= N_ACTION:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((arr.shape[0], N_ACTION), dtype=np.float32)
    out[:, :arr.shape[1]] = arr
    return out


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray,
                  labels: List[int], n_total: int) -> float:
    """Macro F1 via single-pass bincount confusion matrix. ~3-25x faster than
    sklearn f1_score on this workload. Verified equivalent for y_true ranges
    contained in the labels list."""
    cm = np.bincount(y_true.astype(np.int64) * n_total + y_pred.astype(np.int64),
                     minlength=n_total * n_total).reshape(n_total, n_total)
    col_sum = cm.sum(axis=0)
    row_sum = cm.sum(axis=1)
    diag = np.diag(cm)
    f1s = np.zeros(len(labels), dtype=np.float64)
    for i, c in enumerate(labels):
        tp = diag[c]; fp = col_sum[c] - tp; fn = row_sum[c] - tp
        denom = 2 * tp + fp + fn
        if denom <= 0:
            f1s[i] = 0.0
        else:
            f1s[i] = (2 * tp) / denom
    return float(f1s.mean())


def load_components(tags: List[str]) -> Dict:
    """Load OOF + test artifacts for the given tags. Run hard alignment checks."""
    ref = "v16_testhist_aug"
    if not os.path.exists(f"{OOF_DIR}/{ref}_oof_y_act.npy"):
        raise FileNotFoundError(f"Reference tag {ref} OOF y arrays missing")
    y_a = np.load(f"{OOF_DIR}/{ref}_oof_y_act.npy")
    y_p = np.load(f"{OOF_DIR}/{ref}_oof_y_pt.npy")
    y_s = np.load(f"{OOF_DIR}/{ref}_oof_y_srv.npy")
    nsn = np.load(f"{OOF_DIR}/{ref}_oof_nsn.npy")
    mask = np.load(f"{OOF_DIR}/{ref}_oof_mask.npy")
    test_uid = np.load(f"{OOF_DIR}/{ref}_test_rally_uid.npy")

    if int(mask.sum()) != len(y_a):
        raise AssertionError(
            f"Reference OOF mask sum {int(mask.sum())} != y len {len(y_a)}")

    comp: Dict[str, Dict[str, np.ndarray]] = {}
    for tag in tags:
        d: Dict[str, np.ndarray] = {}
        d["oof_act"] = pad_act19(np.load(f"{OOF_DIR}/{tag}_oof_act.npy"))
        d["oof_pt"] = np.load(f"{OOF_DIR}/{tag}_oof_pt.npy").astype(np.float32, copy=False)
        d["oof_srv"] = np.load(f"{OOF_DIR}/{tag}_oof_srv.npy").astype(np.float32, copy=False)
        d["test_act"] = pad_act19(np.load(f"{OOF_DIR}/{tag}_test_act.npy"))
        d["test_pt"] = np.load(f"{OOF_DIR}/{tag}_test_pt.npy").astype(np.float32, copy=False)
        d["test_srv"] = np.load(f"{OOF_DIR}/{tag}_test_srv.npy").astype(np.float32, copy=False)

        # Hard check 1: tag-specific OOF mask matches reference.
        tag_mask_path = f"{OOF_DIR}/{tag}_oof_mask.npy"
        if os.path.exists(tag_mask_path):
            tag_mask = np.load(tag_mask_path)
            if not np.array_equal(tag_mask, mask):
                raise AssertionError(f"OOF mask mismatch for tag {tag}")

        # Hard check 2: tag-specific OOF y arrays match reference (where present).
        for suf, ref_arr, name in [
            ("oof_y_act", y_a, "y_a"),
            ("oof_y_pt",  y_p, "y_p"),
            ("oof_y_srv", y_s, "y_s"),
        ]:
            path = f"{OOF_DIR}/{tag}_{suf}.npy"
            if os.path.exists(path):
                arr = np.load(path)
                if not np.array_equal(arr, ref_arr):
                    raise AssertionError(f"OOF {name} mismatch for tag {tag}")

        # Hard check 3: test rally_uid matches reference (or load v11 from CSV).
        tag_uid_path = f"{OOF_DIR}/{tag}_test_rally_uid.npy"
        if os.path.exists(tag_uid_path):
            tag_uid = np.load(tag_uid_path)
            if not np.array_equal(tag_uid, test_uid):
                raise AssertionError(f"Test rally_uid mismatch for tag {tag}")
        elif tag == "v11":
            v11_sub_path = os.path.join(SUBMISSION_DIR, "submission_v11_transformer.csv")
            v11_uid = pd.read_csv(v11_sub_path)["rally_uid"].values
            if not np.array_equal(v11_uid, test_uid):
                raise AssertionError(
                    "v11 test rally_uid (from submission CSV) does not match reference order")
        else:
            raise AssertionError(f"Test rally_uid file missing for tag {tag}")

        # Hard check 4: probability arrays are finite.
        for nm, arr in d.items():
            if not np.isfinite(arr).all():
                raise AssertionError(f"Non-finite values in {tag}_{nm}")

        comp[tag] = d

    return {
        "comp": comp,
        "y_a": y_a, "y_p": y_p, "y_s": y_s,
        "nsn": nsn, "mask": mask, "test_uid": test_uid,
    }


# ---------- Random-search blends ----------

def _draw_weights(rng, n: int, anchor: np.ndarray = None, alpha: float = 1.0):
    """Sample a length-n simplex vector. Default Dirichlet(1). When `anchor` is
    given, sample as `(1-α)·anchor + α·Dirichlet(1)`, which lies in the
    convex combination of anchor and a random simplex point. L1 distance from
    anchor is bounded by `2·α` (since both vectors sum to 1).

    Output is renormalised to sum to exactly 1.0 (anchor strings from a prior
    ranking CSV are 3-decimal-rounded and may not sum exactly to 1; the
    convex combination then drifts slightly off the simplex).
    """
    base = rng.dirichlet(np.ones(n))
    if anchor is None:
        return base
    out = (1.0 - alpha) * anchor + alpha * base
    return out / out.sum()


def random_search_action(probs_list: List[np.ndarray], y, n_samples, rng,
                         anchor=None, anchor_alpha=1.0):
    n = len(probs_list)
    stack = np.stack(probs_list, axis=0)
    best_w, best_f1 = None, -1.0
    # Always evaluate the anchor itself first when anchored search is on.
    if anchor is not None:
        anchor = anchor / anchor.sum()  # renormalise (CSV strings may sum to 0.999...)
        blend0 = (anchor[:, None, None] * stack).sum(axis=0)
        f0 = fast_macro_f1(y, blend0.argmax(axis=1), ACTION_EVAL, N_ACTION)
        best_f1, best_w = f0, anchor
    for _ in range(n_samples):
        w = _draw_weights(rng, n, anchor=anchor, alpha=anchor_alpha)
        blend = (w[:, None, None] * stack).sum(axis=0)
        f = fast_macro_f1(y, blend.argmax(axis=1), ACTION_EVAL, N_ACTION)
        if f > best_f1:
            best_f1, best_w = f, w
    return best_w, best_f1


def random_search_point(probs_list: List[np.ndarray], y, n_samples, rng,
                        anchor=None, anchor_alpha=1.0):
    n = len(probs_list)
    stack = np.stack(probs_list, axis=0)
    best_w, best_f1 = None, -1.0
    if anchor is not None:
        anchor = anchor / anchor.sum()
        blend0 = (anchor[:, None, None] * stack).sum(axis=0)
        f0 = fast_macro_f1(y, blend0.argmax(axis=1), POINT_EVAL, N_POINT)
        best_f1, best_w = f0, anchor
    for _ in range(n_samples):
        w = _draw_weights(rng, n, anchor=anchor, alpha=anchor_alpha)
        blend = (w[:, None, None] * stack).sum(axis=0)
        f = fast_macro_f1(y, blend.argmax(axis=1), POINT_EVAL, N_POINT)
        if f > best_f1:
            best_f1, best_w = f, w
    return best_w, best_f1


def random_search_server(probs_list: List[np.ndarray], y, n_samples, rng,
                         anchor=None, anchor_alpha=1.0):
    n = len(probs_list)
    stack = np.stack(probs_list, axis=0)
    best_w, best_auc = None, -1.0
    if anchor is not None:
        anchor = anchor / anchor.sum()
        blend0 = (anchor[:, None] * stack).sum(axis=0)
        a0 = roc_auc_score(y, blend0)
        best_auc, best_w = a0, anchor
    for _ in range(n_samples):
        w = _draw_weights(rng, n, anchor=anchor, alpha=anchor_alpha)
        blend = (w[:, None] * stack).sum(axis=0)
        auc = roc_auc_score(y, blend)
        if auc > best_auc:
            best_auc, best_w = auc, w
    return best_w, best_auc


# ---------- Calibration variants ----------

def calib_thr(probs, y, labels, init_cw, n_classes, temp_min=0.5):
    """Full path: temperature search + greedy CW + scipy Powell.

    `temp_min` lower-bounds the temperature grid (default 0.5; P1.5 uses 0.3 to
    test whether grid-edge selection in zoo_v2/zoo_v3 was OOF-overfit).
    """
    best_t, best_f1 = 1.0, -1.0
    for t in np.arange(temp_min, 3.55, 0.1):
        scaled = probs ** (1.0 / t)
        scaled = scaled / scaled.sum(axis=1, keepdims=True)
        f = fast_macro_f1(y, scaled.argmax(axis=1), labels, n_classes)
        if f > best_f1:
            best_f1, best_t = f, t
    probs_t = probs ** (1.0 / best_t)
    probs_t /= probs_t.sum(axis=1, keepdims=True)

    w = np.array([init_cw.get(c, 1.0) for c in range(n_classes)])
    cur_f1 = fast_macro_f1(y, (probs_t * w).argmax(axis=1), labels, n_classes)
    grid = np.concatenate([np.arange(0.05, 1.0, 0.1), np.arange(1.0, 40.0, 1.0)])
    for c in range(n_classes):
        best_wc, best_local = w[c], cur_f1
        for wc in grid:
            trial = w.copy(); trial[c] = wc
            f = fast_macro_f1(y, (probs_t * trial).argmax(axis=1), labels, n_classes)
            if f > best_local:
                best_local, best_wc = f, wc
        w[c] = best_wc; cur_f1 = best_local

    def neg_f1(log_w):
        ww = np.exp(np.clip(log_w, -5, 5))
        return -fast_macro_f1(y, (probs_t * ww).argmax(axis=1), labels, n_classes)
    try:
        res = minimize(neg_f1, np.log(np.clip(w, 0.01, 100)),
                       method="Powell", options={"maxiter": 100})
        if -res.fun > cur_f1:
            w = np.exp(np.clip(res.x, -5, 5))
            cur_f1 = -res.fun
    except Exception:
        pass
    return float(best_t), w, probs_t, float(cur_f1)


def calib_temp(probs, y, labels, n_classes, temp_min=0.5):
    best_t, best_f1 = 1.0, -1.0
    for t in np.arange(temp_min, 3.55, 0.1):
        scaled = probs ** (1.0 / t)
        scaled = scaled / scaled.sum(axis=1, keepdims=True)
        f = fast_macro_f1(y, scaled.argmax(axis=1), labels, n_classes)
        if f > best_f1:
            best_f1, best_t = f, t
    out = probs ** (1.0 / best_t)
    out /= out.sum(axis=1, keepdims=True)
    return float(best_t), np.ones(n_classes), out, float(best_f1)


def calib_cw(probs, y, labels, init_cw, n_classes):
    w = np.array([init_cw.get(c, 1.0) for c in range(n_classes)])
    f = fast_macro_f1(y, (probs * w).argmax(axis=1), labels, n_classes)
    return 1.0, w, probs, float(f)


def calib_none(probs, y, labels, n_classes):
    f = fast_macro_f1(y, probs.argmax(axis=1), labels, n_classes)
    return 1.0, np.ones(n_classes), probs, float(f)


# ---------- Per-SN spread ----------

def per_sn_bucket_ov(pred_a, pred_p, blend_srv, y_a, y_p, y_s, nsn):
    rows = []
    ovs = []
    for name, fn in SN_BUCKETS:
        mask = fn(nsn)
        n = int(mask.sum())
        if n == 0:
            rows.append((name, 0.0, 0))
            ovs.append(0.0)
            continue
        f_a = fast_macro_f1(y_a[mask], pred_a[mask], ACTION_EVAL, N_ACTION)
        f_p = fast_macro_f1(y_p[mask], pred_p[mask], POINT_EVAL, N_POINT)
        if len(np.unique(y_s[mask])) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(y_s[mask], blend_srv[mask])
        ov = 0.4 * f_a + 0.4 * f_p + 0.2 * auc
        rows.append((name, ov, n))
        ovs.append(ov)
    spread = max(ovs) - min(ovs)
    return rows, float(spread)


# ---------- Subset enumeration ----------

def enumerate_subsets(max_models: int = 6):
    A_choices = [[]] + [[g] for g in GROUP_A]
    B_choices = [[]] + [[g] for g in GROUP_B]
    C_choices = [[]] + [[g] for g in GROUP_C]
    # Group D: at least 1, up to len(GROUP_D) — bounded by max_models in the size filter.
    D_choices = []
    for r in range(1, len(GROUP_D) + 1):
        for combo in itertools.combinations(GROUP_D, r):
            D_choices.append(list(combo))
    E_choices = [[]] + [[g] for g in GROUP_E]
    seen = set()
    for a in A_choices:
        for b in B_choices:
            for c in C_choices:
                for d in D_choices:
                    for e in E_choices:
                        subset = a + b + c + d + e
                        if 3 <= len(subset) <= max_models:
                            key = tuple(sorted(subset))
                            if key not in seen:
                                seen.add(key)
                                yield list(key)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=300,
                    help="Random-search Dirichlet draws per task.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="Random-search seed (fixed for reproducibility).")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Number of top submissions to materialise (counted within "
                         "eligible candidates, see --edge-cushion).")
    ap.add_argument("--max-models", type=int, default=6,
                    help="Cap blend size (TRAIN_PLAN P1.5 sets this to 5 after "
                         "zoo_v3 size-6 LB regression).")
    ap.add_argument("--temp-min", type=float, default=0.5,
                    help="Lower bound for THR/TEMP temperature search grid. P1.5 "
                         "sets this to 0.3 to test the zoo_v2/zoo_v3 grid-edge bias.")
    ap.add_argument("--edge-cushion", type=float, default=0.05,
                    help="A candidate is 'edge' iff min(t_a, t_p) <= temp_min + cushion. "
                         "Edge candidates are excluded from the eligible top-K materialisation.")
    ap.add_argument("--ranking-out",
                    default=os.path.join(SUBMISSION_DIR, "zoo_v2_ranking.csv"))
    ap.add_argument("--prefix", default="zoo_v2",
                    help="Submission filename prefix.")
    ap.add_argument("--replace", default=None,
                    help="Comma-separated old:new (e.g. v16_testhist_aug:v16_avg3).")
    ap.add_argument("--anchor-from", default=None,
                    help="Path to a prior ranking CSV (e.g. submissions/zoo_v2_ranking.csv) "
                         "to anchor the search around. P12: weight perturbation around the "
                         "LB-validated zoo_v2 top-1.")
    ap.add_argument("--anchor-rank", type=int, default=1,
                    help="Which rank in the anchor ranking CSV to use (default 1 = top-1).")
    ap.add_argument("--anchor-alpha", type=float, default=0.1,
                    help="Convex-combination weight for perturbation: "
                         "w_new = (1-α)*anchor + α*Dirichlet. L1 drift bounded by 2α. "
                         "Default 0.1 → max L1 drift 0.2.")
    args = ap.parse_args()

    print("=== blend_zoo_v2 - purpose-built N-way blender ===")
    print(f"seed={args.seed}  n_samples={args.n_samples}  top_k={args.top_k}  "
          f"max_models={args.max_models}  temp_min={args.temp_min}  "
          f"edge_cushion={args.edge_cushion}")
    rng = np.random.default_rng(args.seed)

    replace_map: Dict[str, str] = {}
    if args.replace:
        for tok in args.replace.split(","):
            old, new = tok.split(":")
            replace_map[old.strip()] = new.strip()
    if replace_map:
        print(f"replace_map={replace_map}")

    def remap(t: str) -> str:
        return replace_map.get(t, t)

    used_tags = sorted({remap(t) for t in ALL_TAGS})
    data = load_components(used_tags)
    comp = data["comp"]
    y_a, y_p, y_s = data["y_a"], data["y_p"], data["y_s"]
    nsn, mask, test_uid = data["nsn"], data["mask"], data["test_uid"]
    print(f"Loaded {len(used_tags)} tags. OOF n={len(y_a)} (mask sum={int(mask.sum())}).")
    print(f"Test n={len(test_uid)}.")
    print(f"All hard alignment checks passed.")

    # P12 anchor mode: restrict the search to the anchor's exact subset and
    # sample weights as convex combinations of (anchor, fresh Dirichlet).
    anchor_per_subset: Dict[Tuple[str, ...], Dict[str, np.ndarray]] = {}
    if args.anchor_from:
        print(f"\n=== P12 anchor mode ===")
        print(f"  anchor_from={args.anchor_from}  rank={args.anchor_rank}  alpha={args.anchor_alpha}")
        anc_df = pd.read_csv(args.anchor_from)
        anc = anc_df[anc_df["rank"] == args.anchor_rank].iloc[0]
        anc_tags = anc["subset"].split("+")
        anc_w_a = np.array([float(x) for x in anc["w_a"].split(",")])
        anc_w_p = np.array([float(x) for x in anc["w_p"].split(",")])
        anc_w_s = np.array([float(x) for x in anc["w_s"].split(",")])
        # Reorder weights so they match sorted(set(remap(tags))) — the canonical
        # enumeration order used by the search loop.
        used_anchor = sorted(set(remap(t) for t in anc_tags))
        order = [anc_tags.index(t) for t in used_anchor]
        anchor_per_subset[tuple(used_anchor)] = {
            "w_a": anc_w_a[order], "w_p": anc_w_p[order], "w_s": anc_w_s[order],
        }
        print(f"  anchor subset (sorted): {used_anchor}")
        print(f"  anchor weights aligned to sorted order; search restricted to this subset only.")

    raw_subsets = list(enumerate_subsets(max_models=args.max_models))
    seen = set()
    unique_subsets: List[Tuple[List[str], List[str]]] = []
    for orig in raw_subsets:
        used = sorted(set(remap(t) for t in orig))
        if not (3 <= len(used) <= 6):
            continue
        key = tuple(used)
        if key not in seen:
            seen.add(key)
            unique_subsets.append((orig, used))

    # Anchor mode: keep only the anchor's subset (other subsets get no perturbation).
    if anchor_per_subset:
        anchor_keys = set(anchor_per_subset.keys())
        unique_subsets = [(o, u) for (o, u) in unique_subsets if tuple(u) in anchor_keys]
        if not unique_subsets:
            raise AssertionError(
                f"P12 anchor subset {list(anchor_per_subset.keys())[0]} not in enumeration "
                "(check --max-models / --replace consistency).")
        print(f"  P12: search restricted to {len(unique_subsets)} subset(s).")
    print(f"Enumerated {len(unique_subsets)} unique subsets after remap and de-dupe.")

    rows: List[Dict] = []
    t_start = time.time()
    for i, (sub_orig, sub_used) in enumerate(unique_subsets):
        t0 = time.time()
        n_models = len(sub_used)
        oof_acts = [comp[t]["oof_act"] for t in sub_used]
        oof_pts = [comp[t]["oof_pt"] for t in sub_used]
        oof_srvs = [comp[t]["oof_srv"] for t in sub_used]

        # P12 anchor mode: pull per-task anchor weights for this subset, if any.
        anc = anchor_per_subset.get(tuple(sub_used), None)
        anc_a = anc["w_a"] if anc is not None else None
        anc_p = anc["w_p"] if anc is not None else None
        anc_s = anc["w_s"] if anc is not None else None
        a_alpha = args.anchor_alpha if anc is not None else 1.0

        w_a, raw_f1_a = random_search_action(oof_acts, y_a, args.n_samples, rng,
                                              anchor=anc_a, anchor_alpha=a_alpha)
        w_p, raw_f1_p = random_search_point(oof_pts, y_p, args.n_samples, rng,
                                             anchor=anc_p, anchor_alpha=a_alpha)
        w_s, raw_auc = random_search_server(oof_srvs, y_s, args.n_samples, rng,
                                             anchor=anc_s, anchor_alpha=a_alpha)

        # Hard check: weight vectors sum to 1.
        for nm, w in [("w_a", w_a), ("w_p", w_p), ("w_s", w_s)]:
            if abs(float(w.sum()) - 1.0) > 1e-6:
                raise AssertionError(
                    f"{nm} for subset {sub_used} does not sum to 1: {w.sum()}")

        stack_a = np.stack(oof_acts, axis=0)
        stack_p = np.stack(oof_pts, axis=0)
        stack_s = np.stack(oof_srvs, axis=0)
        blend_a = (w_a[:, None, None] * stack_a).sum(axis=0)
        blend_p = (w_p[:, None, None] * stack_p).sum(axis=0)
        blend_s = (w_s[:, None] * stack_s).sum(axis=0)

        for nm, arr in [("blend_a", blend_a), ("blend_p", blend_p), ("blend_s", blend_s)]:
            if not np.isfinite(arr).all():
                raise AssertionError(f"NaN/Inf in {nm} for subset {sub_used}")

        for calib in ["THR", "TEMP", "CW", "NONE"]:
            if calib == "THR":
                t_a, ww_a, pa_cal, f1_a = calib_thr(blend_a, y_a, ACTION_EVAL, ACTION_CW, N_ACTION, temp_min=args.temp_min)
                t_p, ww_p, pp_cal, f1_p = calib_thr(blend_p, y_p, POINT_EVAL, POINT_CW, N_POINT, temp_min=args.temp_min)
            elif calib == "TEMP":
                t_a, ww_a, pa_cal, f1_a = calib_temp(blend_a, y_a, ACTION_EVAL, N_ACTION, temp_min=args.temp_min)
                t_p, ww_p, pp_cal, f1_p = calib_temp(blend_p, y_p, POINT_EVAL, N_POINT, temp_min=args.temp_min)
            elif calib == "CW":
                t_a, ww_a, pa_cal, f1_a = calib_cw(blend_a, y_a, ACTION_EVAL, ACTION_CW, N_ACTION)
                t_p, ww_p, pp_cal, f1_p = calib_cw(blend_p, y_p, POINT_EVAL, POINT_CW, N_POINT)
            else:  # NONE
                t_a, ww_a, pa_cal, f1_a = calib_none(blend_a, y_a, ACTION_EVAL, N_ACTION)
                t_p, ww_p, pp_cal, f1_p = calib_none(blend_p, y_p, POINT_EVAL, N_POINT)

            auc = float(roc_auc_score(y_s, blend_s))
            ov = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc

            pred_a = (pa_cal * ww_a).argmax(axis=1)
            pred_p = (pp_cal * ww_p).argmax(axis=1)
            sn_rows, sn_spread = per_sn_bucket_ov(
                pred_a, pred_p, blend_s, y_a, y_p, y_s, nsn)

            rows.append({
                "subset": "+".join(sub_used),
                "n_models": n_models,
                "calibration": calib,
                "w_a": ",".join(f"{x:.3f}" for x in w_a),
                "w_p": ",".join(f"{x:.3f}" for x in w_p),
                "w_s": ",".join(f"{x:.3f}" for x in w_s),
                "t_a": float(t_a), "t_p": float(t_p),
                "f1_a": float(f1_a), "f1_p": float(f1_p), "auc": float(auc),
                "oof_ov": float(ov),
                "sn_spread": float(sn_spread),
                "sn_buckets": ";".join(f"{n}={v:.4f}" for n, v, _ in sn_rows),
                "_w_a_arr": w_a, "_w_p_arr": w_p, "_w_s_arr": w_s,
                "_t_a": t_a, "_t_p": t_p,
                "_ww_a_arr": ww_a, "_ww_p_arr": ww_p,
                "_sub_used": tuple(sub_used),
            })

        elapsed = time.time() - t0
        raw_ov = 0.4 * raw_f1_a + 0.4 * raw_f1_p + 0.2 * raw_auc
        eta_s = elapsed * (len(unique_subsets) - i - 1)
        print(f"[{i+1:>3}/{len(unique_subsets)}] {sub_used} (n={n_models}) "
              f"raw_OV={raw_ov:.4f}  elapsed={elapsed:.1f}s  ETA={eta_s/60.0:.1f}m")

    print(f"\nSearch complete. Total elapsed = {(time.time() - t_start)/60.0:.1f} min")

    ref_used_sorted = tuple(sorted(set(remap(t) for t in REF_SUBSET)))
    ref_rows = [r for r in rows
                if r["_sub_used"] == ref_used_sorted and r["calibration"] == REF_CALIB]
    if ref_rows:
        ref_spread = ref_rows[0]["sn_spread"]
        ref_ov = ref_rows[0]["oof_ov"]
        print(f"Reference subset {ref_used_sorted} ({REF_CALIB}) found.")
        print(f"   ref_oof_ov = {ref_ov:.4f}   ref_sn_spread = {ref_spread:.4f}")
    else:
        ref_spread = float(np.median([r["sn_spread"] for r in rows]))
        ref_ov = float("nan")
        print(f"Reference subset NOT found (likely under --replace).")
        print(f"   Using median spread {ref_spread:.4f} as fallback reference.")

    for r in rows:
        r["spread_penalised_score"] = (
            r["oof_ov"] - 0.5 * max(0.0, r["sn_spread"] - ref_spread))
        # Edge-rejection annotation (Codex 2026-05-05): TEMP/THR candidates whose
        # chosen temperature lies on the lower edge of the search grid are suspect.
        # CW and NONE never set t_a/t_p (they're 1.0 by construction), so flag only
        # TEMP/THR variants here.
        if r["calibration"] in ("THR", "TEMP"):
            r["temp_at_edge"] = (
                min(r["_t_a"], r["_t_p"]) <= args.temp_min + args.edge_cushion)
        else:
            r["temp_at_edge"] = False

    rows.sort(key=lambda r: r["spread_penalised_score"], reverse=True)
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank

    # Eligible rank: rank among non-edge candidates only. (Codex 2026-05-05)
    eligible_idx = 0
    for r in rows:
        if r["temp_at_edge"]:
            r["eligible_rank"] = ""  # empty cell in CSV (NaN-equivalent)
        else:
            eligible_idx += 1
            r["eligible_rank"] = eligible_idx

    # Hard check (post-sort): all weight vectors sum to 1.
    for r in rows:
        for nm in ("_w_a_arr", "_w_p_arr", "_w_s_arr"):
            s = float(r[nm].sum())
            if abs(s - 1.0) > 1e-6:
                raise AssertionError(
                    f"Weight {nm} for {r['subset']} ({r['calibration']}) does not sum to 1: {s}")

    # Pre-write the ranking CSV (without filenames yet).
    keep_cols = [
        "rank", "eligible_rank", "subset", "n_models", "calibration",
        "w_a", "w_p", "w_s", "t_a", "t_p", "temp_at_edge",
        "f1_a", "f1_p", "auc", "oof_ov",
        "sn_spread", "sn_buckets", "spread_penalised_score",
    ]
    df_records = [{k: r.get(k, "") for k in keep_cols} for r in rows]
    df = pd.DataFrame(df_records)
    df["file"] = ""

    # Print global top-K (informational; some may be edge-rejected).
    print(f"\n=== Global top-{args.top_k} candidates (by spread_penalised_score) ===")
    print(f"{'rank':>4}  {'elig':>4}  {'calib':<5}  {'oof_ov':>7}  {'spread':>7}  "
          f"{'sps':>7}  {'edge':<5}  {'subset'}")
    for r in rows[:args.top_k]:
        elig = r["eligible_rank"] if r["eligible_rank"] != "" else "—"
        print(f"{r['rank']:>4}  {str(elig):>4}  {r['calibration']:<5}  "
              f"{r['oof_ov']:>7.4f}  {r['sn_spread']:>7.4f}  "
              f"{r['spread_penalised_score']:>7.4f}  "
              f"{'YES' if r['temp_at_edge'] else 'no':<5}  {r['subset']}")

    # Materialise the top-K *eligible* rows as submission CSVs (Codex 2026-05-05).
    eligible_rows = [r for r in rows if not r["temp_at_edge"]][:args.top_k]
    print(f"\n=== Eligible top-{args.top_k} candidates "
          f"(temp interior, materialised as submissions) ===")
    print(f"{'elig':>4}  {'rank':>4}  {'calib':<5}  {'oof_ov':>7}  {'spread':>7}  "
          f"{'sps':>7}  {'subset'}")
    for r in eligible_rows:
        print(f"{r['eligible_rank']:>4}  {r['rank']:>4}  {r['calibration']:<5}  "
              f"{r['oof_ov']:>7.4f}  {r['sn_spread']:>7.4f}  "
              f"{r['spread_penalised_score']:>7.4f}  {r['subset']}")

    if not eligible_rows:
        print("  (no eligible candidates — all top-K are edge-rejected!)")

    print()
    for r in eligible_rows:
        sub_used = list(r["_sub_used"])
        test_acts = [comp[t]["test_act"] for t in sub_used]
        test_pts = [comp[t]["test_pt"] for t in sub_used]
        test_srvs = [comp[t]["test_srv"] for t in sub_used]

        ta = (r["_w_a_arr"][:, None, None] * np.stack(test_acts, axis=0)).sum(axis=0)
        tp = (r["_w_p_arr"][:, None, None] * np.stack(test_pts, axis=0)).sum(axis=0)
        ts = (r["_w_s_arr"][:, None] * np.stack(test_srvs, axis=0)).sum(axis=0)

        if r["calibration"] in ("THR", "TEMP"):
            ta_t = ta ** (1.0 / r["_t_a"]); ta_t /= ta_t.sum(axis=1, keepdims=True)
            tp_t = tp ** (1.0 / r["_t_p"]); tp_t /= tp_t.sum(axis=1, keepdims=True)
        else:
            ta_t = ta
            tp_t = tp
        if r["calibration"] in ("THR", "CW"):
            pred_a = (ta_t * r["_ww_a_arr"]).argmax(axis=1)
            pred_p = (tp_t * r["_ww_p_arr"]).argmax(axis=1)
        else:
            pred_a = ta_t.argmax(axis=1)
            pred_p = tp_t.argmax(axis=1)

        for nm, arr in [("test_act", ta_t), ("test_pt", tp_t), ("test_srv", ts)]:
            if not np.isfinite(arr).all():
                raise AssertionError(
                    f"NaN/Inf in {nm} for elig-{r['eligible_rank']} subset {sub_used}")

        prov = "_".join(
            t.replace("v16_testhist_aug", "v16")
             .replace("v14_seed", "v14s")
             .replace("v14_avg3", "v14avg3")
             .replace("v14_5f_nocb", "v14nocb")
             .replace("v12_5f", "v125f")
            for t in sub_used
        )
        fname = (f"submission_{args.prefix}_elig{r['eligible_rank']}_"
                 f"{r['calibration'].lower()}_{prov}.csv")
        out_path = os.path.join(SUBMISSION_DIR, fname)
        sub = pd.DataFrame({
            "rally_uid": test_uid,
            "actionId": pred_a,
            "pointId": pred_p,
            "serverGetPoint": ts,
        })
        sub.to_csv(out_path, index=False, lineterminator="\n")
        df.loc[df["rank"] == r["rank"], "file"] = fname
        print(f"  elig={r['eligible_rank']:>2} (global rank {r['rank']:>3})  ->  {fname}")

    df.to_csv(args.ranking_out, index=False, lineterminator="\n")
    print(f"\nSaved ranking: {args.ranking_out}  ({len(df)} entries, "
          f"{sum(1 for r in rows if not r['temp_at_edge'])} eligible)")


if __name__ == "__main__":
    main()
