"""R-065c expanded teacher-pool audit (NO TRAINING).

Per Codex BLOCK on R-065b (2026-05-23):
- Cannot use both v11_aug_oldtest and v11_aug_oldtest_avg3 (numerically identical).
- GBM cluster still highly correlated; treat as a single collapsed teacher.
- Need genuinely distinct teachers — search broader transformer pool.

This script:
1. Loads all non-toxic transformer candidates + collapses the GBM cluster.
2. Reports pairwise PROB CORRELATION (Codex's binding gate) for action / point / srv.
3. Identifies "distinct" teachers (prob corr < 0.85 with all selected peers).
4. Greedy-picks distinct transformers to pair with the GBM cluster.
5. Writes `submissions/r065c_teacher_pool_audit.json` with the full matrices +
   the recommended teacher set for the v2c generator.

USAGE:
    python -u src/audit_teacher_pool_v2c.py
"""
import hashlib
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT  # noqa: E402

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
N_ACT_FOR_CORR = 15
CORRELATION_BLOCK_THRESHOLD = 0.85

# GBM cluster (4 teachers with mutual prob corr 0.93-0.98 — collapse to 1 vote).
GBM_CLUSTER = [
    "v14_seed2_v15feat_a",
    "v13_oldtest",
    "v16_testhist_aug_oldtest",
    "v14_seed2_v16match_v2",
]
GBM_CLUSTER_TAG = "gbm_cluster"

# Transformer candidates (excluding mulminet family — LB-toxic per R-040/R-055).
# Also exclude v11_aug_oldtest_avg3 (Codex verified identical to v11_aug_oldtest).
TRANSFORMER_CANDIDATES = [
    "v11_aug_oldtest",       # baseline transformer
    "v11plus",               # different transformer family
    "v11plus_oldtest",       # v11plus + oldtest
    "v11plus_oldtest_avg2",  # v11plus + oldtest avg2
    "v11_aug",               # transformer, no oldtest
    "v11",                   # baseline-baseline transformer
    "v11_uncertainty_aug",   # uncertainty-trained variant
]


def load_one(tag: str, n_canonical: int) -> Dict[str, np.ndarray]:
    out = {}
    for suffix in ["oof_act", "oof_pt", "oof_srv", "test_act", "test_pt", "test_srv",
                   "test_rally_uid"]:
        path = os.path.join(OOF_DIR, f"{tag}_{suffix}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"MISSING {path}")
        arr = np.load(path)
        if suffix.startswith("oof_") and arr.shape[0] > n_canonical:
            arr = arr[:n_canonical]
        out[suffix] = arr
    return out


def slice_to_15(act: np.ndarray) -> np.ndarray:
    if act.shape[1] < N_ACT_FOR_CORR:
        return act.astype(np.float32)
    sliced = act[:, :N_ACT_FOR_CORR].astype(np.float64)
    s = sliced.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return (sliced / s).astype(np.float32)


def prob_corr(a: np.ndarray, b: np.ndarray) -> float:
    flat_a = a.astype(np.float64).flatten()
    flat_b = b.astype(np.float64).flatten()
    if flat_a.std() == 0 or flat_b.std() == 0:
        return 0.0
    return float(np.corrcoef(flat_a, flat_b)[0, 1])


def collapse_gbm_cluster(cluster_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Average probabilities across the GBM cluster members. Treated as ONE teacher."""
    tags = list(cluster_data.keys())
    out = {}
    for suffix in ["oof_act", "oof_pt", "oof_srv", "test_act", "test_pt", "test_srv"]:
        stack = np.stack([cluster_data[t][suffix].astype(np.float64) for t in tags], axis=0)
        mean = stack.mean(axis=0)
        # Normalize prob arrays (action/point) row-wise to ensure they sum to 1
        if suffix in ("oof_act", "oof_pt", "test_act", "test_pt"):
            s = mean.sum(axis=1, keepdims=True)
            s = np.where(s == 0, 1.0, s)
            mean = mean / s
        out[suffix] = mean.astype(np.float32)
    out["test_rally_uid"] = cluster_data[tags[0]]["test_rally_uid"]
    return out


def hash_uids(uids: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(uids).tobytes())
    return h.hexdigest()[:16]


def greedy_select_distinct(
    candidate_probs: Dict[str, np.ndarray],
    must_include: List[str],
    threshold: float,
) -> Tuple[List[str], List[str]]:
    """Greedy pick: start with must_include; add each candidate that has prob corr
    < threshold with ALL already-selected teachers.

    Returns (selected, rejected) tag lists.
    """
    selected = list(must_include)
    rejected: List[str] = []
    # Score-order the rest by distinctness from must_include (lowest mean corr first)
    rest = [t for t in candidate_probs if t not in selected]
    if must_include:
        avg_corr_to_must = {
            t: float(np.mean([prob_corr(candidate_probs[t], candidate_probs[s])
                              for s in must_include]))
            for t in rest
        }
        rest_sorted = sorted(rest, key=lambda t: avg_corr_to_must[t])
    else:
        rest_sorted = rest

    for t in rest_sorted:
        if all(prob_corr(candidate_probs[t], candidate_probs[s]) < threshold for s in selected):
            selected.append(t)
        else:
            rejected.append(t)
    return selected, rejected


def main() -> None:
    print("=" * 78)
    print(" R-065c — Cluster-aware Teacher Pool Audit (NO TRAINING)")
    print(" Codex R-065b BLOCK 2026-05-23")
    print("=" * 78)

    # Reference canonical OOF size
    ref = load_one("v14_seed2_v15feat_a", 999_999)
    n_canonical = ref["oof_act"].shape[0]
    ref_test_uid = ref["test_rally_uid"]
    uid_hash = hash_uids(ref_test_uid)
    print(f" canonical OOF: {n_canonical}  test_uid_hash: {uid_hash}")

    # Load GBM cluster members
    cluster_data: Dict[str, Dict[str, np.ndarray]] = {}
    for tag in GBM_CLUSTER:
        cluster_data[tag] = load_one(tag, n_canonical)
        print(f"  loaded GBM {tag}: oof_act={cluster_data[tag]['oof_act'].shape}")
    gbm_collapsed = collapse_gbm_cluster(cluster_data)
    print(f" GBM cluster collapsed (mean of {len(GBM_CLUSTER)} teachers).")

    # Load transformer candidates
    txf_data: Dict[str, Dict[str, np.ndarray]] = {}
    for tag in TRANSFORMER_CANDIDATES:
        try:
            txf_data[tag] = load_one(tag, n_canonical)
            print(f"  loaded transformer {tag}: oof_act={txf_data[tag]['oof_act'].shape}")
        except FileNotFoundError as e:
            print(f"  skip {tag}: {e}")

    # ─── Build full pool: 1 GBM cluster + N transformers ─────────────────────
    pool = {GBM_CLUSTER_TAG: gbm_collapsed, **txf_data}

    # Action: slice all to 15-class
    pool_act = {t: slice_to_15(d["oof_act"]) for t, d in pool.items()}
    pool_pt = {t: d["oof_pt"].astype(np.float32) for t, d in pool.items()}
    pool_srv = {t: d["oof_srv"].astype(np.float64) for t, d in pool.items()}

    print("\n" + "=" * 78)
    print(" PAIRWISE PROBABILITY CORRELATIONS (action, 15-class sliced)")
    print("=" * 78)
    tags = list(pool_act.keys())
    n = len(tags)
    act_corr = np.zeros((n, n))
    for i, ta in enumerate(tags):
        for j, tb in enumerate(tags):
            act_corr[i, j] = prob_corr(pool_act[ta], pool_act[tb])
    act_corr_df = pd.DataFrame(act_corr, index=tags, columns=tags)
    print(act_corr_df.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 78)
    print(" PAIRWISE PROB CORR (point, 10-class)")
    print("=" * 78)
    pt_corr = np.zeros((n, n))
    for i, ta in enumerate(tags):
        for j, tb in enumerate(tags):
            pt_corr[i, j] = prob_corr(pool_pt[ta], pool_pt[tb])
    pt_corr_df = pd.DataFrame(pt_corr, index=tags, columns=tags)
    print(pt_corr_df.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 78)
    print(" PAIRWISE PROB CORR (server, binary)")
    print("=" * 78)
    srv_corr = np.zeros((n, n))
    for i, ta in enumerate(tags):
        for j, tb in enumerate(tags):
            srv_corr[i, j] = prob_corr(pool_srv[ta], pool_srv[tb])
    srv_corr_df = pd.DataFrame(srv_corr, index=tags, columns=tags)
    print(srv_corr_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # ─── Greedy distinct selection ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f" GREEDY DISTINCT TEACHER SELECTION (prob corr < {CORRELATION_BLOCK_THRESHOLD})")
    print(" Must-include: gbm_cluster (collapsed)")
    print(" Selection metric: action prob corr (Codex's binding gate)")
    print("=" * 78)
    selected, rejected = greedy_select_distinct(
        pool_act,
        must_include=[GBM_CLUSTER_TAG],
        threshold=CORRELATION_BLOCK_THRESHOLD,
    )
    print(f"\n SELECTED teachers ({len(selected)}):")
    for t in selected:
        print(f"   {t}")
    print(f"\n REJECTED candidates ({len(rejected)}):")
    for t in rejected:
        max_corr = max(prob_corr(pool_act[t], pool_act[s]) for s in selected)
        print(f"   {t}  (max prob corr with selected = {max_corr:.4f})")

    # ─── Stop-gate decision logic ───────────────────────────────────────────
    print("\n" + "=" * 78)
    print(" STOP GATE DECISION (per user 2026-05-23)")
    print("=" * 78)
    n_distinct_transformers = len(selected) - 1  # subtract gbm_cluster
    print(f"   distinct transformers found: {n_distinct_transformers}")
    print(f"   total consensus votes (incl. GBM cluster): {len(selected)}")
    if len(selected) < 3:
        print("\n   VERDICT: < 3 distinct teachers — INSUFFICIENT for consensus.")
        print("            Per user 2026-05-23: 'If R-065c cannot produce independent")
        print("            teachers plus a valid point pool, abandon consensus pseudo-labeling.'")
        verdict = "abandon_insufficient_teachers"
    elif len(selected) < 4:
        print("\n   VERDICT: 3 votes — minimum viable. Consensus requires 3-of-3 or 2-of-3.")
        verdict = "viable_3vote"
    else:
        print(f"\n   VERDICT: {len(selected)} distinct votes — strong consensus pool.")
        verdict = "viable_strong"

    # ─── Manifest ────────────────────────────────────────────────────────────
    manifest = {
        "stage": "R-065c teacher pool audit",
        "ts": "2026-05-23",
        "canonical_oof_size": int(n_canonical),
        "test_uid_sha256_16": uid_hash,
        "gbm_cluster_members": GBM_CLUSTER,
        "gbm_cluster_collapse_method": "mean_of_probabilities",
        "transformer_candidates": list(txf_data.keys()),
        "correlation_block_threshold": CORRELATION_BLOCK_THRESHOLD,
        "selected_teachers": selected,
        "rejected_teachers": [
            {"tag": t,
             "max_corr_with_selected": float(max(prob_corr(pool_act[t], pool_act[s])
                                                  for s in selected))}
            for t in rejected
        ],
        "verdict": verdict,
        "n_distinct_transformers": n_distinct_transformers,
        "n_total_votes": len(selected),
        "action_prob_corr": act_corr_df.to_dict(),
        "point_prob_corr": pt_corr_df.to_dict(),
        "srv_prob_corr": srv_corr_df.to_dict(),
    }
    out_path = os.path.join(PROJECT_ROOT, "submissions", "r065c_teacher_pool_audit.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, default=float)
    print(f"\n Manifest saved: {out_path}")
    print(f" Recommended teacher set for v2c generator: {selected}")


if __name__ == "__main__":
    main()
