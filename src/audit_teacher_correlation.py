"""R-065 Stage-0 (Codex BLOCK→APPROVE_WITH_FIXES 2026-05-23).

Pairwise OOF correlation audit across 5 candidate teachers for Consensus
Pseudo V2. Required before any pseudo-label generation can proceed.

Codex requirements:
1. Canonical 69712-row OOF alignment (oldtest teachers have 72065-row arrays;
   first 69712 must match canonical labels).
2. Action 15-vs-19 class handling: transformer is 15-class, GBMs are 19-class.
   For correlation, slice GBM to first 15 classes (target shot is non-serve so
   classes 15-18 carry near-zero mass).
3. Report pairwise correlations on action + point + srv tasks.
4. Cluster teachers by correlation (>=0.85 pairs are problematic).
5. Test UID hash for reproducibility manifest.

Output: prints correlation matrices + cluster verdict. Writes JSON manifest to
`submissions/r065_teacher_correlation.json`.

USAGE:
    python -u src/audit_teacher_correlation.py
"""
import hashlib
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT  # noqa: E402

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")

# 5 candidate teachers (per R-065 proposal)
TEACHERS = [
    ("v14_seed2_v15feat_a",       "GBM, R-034 LB-WIN base"),
    ("v11_aug_oldtest",           "Transformer"),
    ("v16_testhist_aug_oldtest",  "GBM + test-history aug"),
    ("v13_oldtest",               "Different GBM hyperparams"),
    ("v14_seed2_v16match_v2",     "NEW LORO features (R-032 v2)"),
]

# Action class slice — first 15 classes only (target shots are non-serve;
# classes 15-18 are serve actions that carry near-zero mass in non-serve prefixes).
N_ACT_FOR_CORR = 15
N_PT = 10
CORRELATION_BLOCK_THRESHOLD = 0.85


def load_oof_block(tag: str, n_canonical: int) -> Dict[str, np.ndarray]:
    """Load OOF arrays + slice to first n_canonical rows for oldtest variants."""
    out: Dict[str, np.ndarray] = {}
    for suffix in ["oof_act", "oof_pt", "oof_srv", "oof_y_act", "oof_y_pt", "oof_y_srv",
                   "test_act", "test_pt", "test_srv", "test_rally_uid"]:
        path = os.path.join(OOF_DIR, f"{tag}_{suffix}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"MISSING: {path}")
        arr = np.load(path)
        # Slice OOF arrays (suffix starts with 'oof_') to canonical N rows.
        if suffix.startswith("oof_") and arr.shape[0] > n_canonical:
            arr = arr[:n_canonical]
        out[suffix] = arr
    return out


def hash_uids(uids: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(uids).tobytes())
    return h.hexdigest()[:16]


def slice_to_15_classes(act: np.ndarray) -> np.ndarray:
    """Slice 19-class action probabilities to first 15 classes, renormalize."""
    if act.shape[1] < N_ACT_FOR_CORR:
        # 15-class transformer: pad to 19 wouldn't help; assume already 15-class.
        return act
    sliced = act[:, :N_ACT_FOR_CORR].astype(np.float64)
    s = sliced.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return (sliced / s).astype(np.float32)


def pairwise_top1_correlation(probs_by_tag: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Correlation matrix of argmax predictions (treats as int-encoded categorical).

    For each pair of teachers, fraction of rows where top-1 prediction matches.
    """
    tags = list(probs_by_tag.keys())
    n = len(tags)
    mat = np.zeros((n, n), dtype=np.float64)
    top1 = {t: probs_by_tag[t].argmax(axis=1) for t in tags}
    n_rows = len(top1[tags[0]])
    for i, ta in enumerate(tags):
        for j, tb in enumerate(tags):
            agree = (top1[ta] == top1[tb]).sum()
            mat[i, j] = agree / n_rows
    return pd.DataFrame(mat, index=tags, columns=tags)


def pairwise_prob_correlation(probs_by_tag: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Pearson correlation of flattened probability arrays (row-wise stacked)."""
    tags = list(probs_by_tag.keys())
    n = len(tags)
    mat = np.zeros((n, n), dtype=np.float64)
    flat = {t: probs_by_tag[t].astype(np.float64).flatten() for t in tags}
    for i, ta in enumerate(tags):
        for j, tb in enumerate(tags):
            mat[i, j] = float(np.corrcoef(flat[ta], flat[tb])[0, 1])
    return pd.DataFrame(mat, index=tags, columns=tags)


def cluster_report(corr: pd.DataFrame, threshold: float) -> List[List[str]]:
    """Greedy clustering: teachers in same cluster have pairwise corr >= threshold.

    Returns list of clusters (each a list of teacher tags).
    """
    tags = list(corr.index)
    assigned = set()
    clusters: List[List[str]] = []
    for t in tags:
        if t in assigned:
            continue
        cluster = [t]
        assigned.add(t)
        for other in tags:
            if other in assigned:
                continue
            if all(corr.loc[member, other] >= threshold for member in cluster):
                cluster.append(other)
                assigned.add(other)
        clusters.append(cluster)
    return clusters


def main() -> None:
    print("=" * 78)
    print(" R-065 Stage-0 — Teacher OOF correlation audit")
    print(" Codex BLOCK→APPROVE_WITH_FIXES 2026-05-23")
    print("=" * 78)

    # Reference: canonical 69712-row OOF from v14_seed2_v15feat_a (no oldtest)
    ref_y_act = np.load(os.path.join(OOF_DIR, "v14_seed2_v15feat_a_oof_y_act.npy"))
    ref_y_pt = np.load(os.path.join(OOF_DIR, "v14_seed2_v15feat_a_oof_y_pt.npy"))
    ref_y_srv = np.load(os.path.join(OOF_DIR, "v14_seed2_v15feat_a_oof_y_srv.npy"))
    ref_test_uid = np.load(os.path.join(OOF_DIR, "v14_seed2_v15feat_a_test_rally_uid.npy"))
    n_canonical = len(ref_y_act)
    uid_hash = hash_uids(ref_test_uid)
    print(f" Canonical OOF size: {n_canonical}  test_uid count: {len(ref_test_uid)}  "
          f"uid_hash: {uid_hash}")

    teacher_data: Dict[str, Dict[str, np.ndarray]] = {}
    for tag, desc in TEACHERS:
        print(f"\n Loading {tag}  ({desc})")
        try:
            d = load_oof_block(tag, n_canonical)
        except FileNotFoundError as e:
            print(f"   SKIP — {e}")
            continue

        # Alignment check
        assert d["oof_y_act"].shape[0] == n_canonical, f"y_act size mismatch for {tag}"
        if not np.array_equal(d["oof_y_act"], ref_y_act):
            print(f"   WARN: oof_y_act differs from reference (sliced); "
                  f"first-mismatch index = {np.argmax(d['oof_y_act'] != ref_y_act)}")
        if not np.array_equal(d["oof_y_pt"], ref_y_pt):
            print(f"   WARN: oof_y_pt differs from reference (sliced)")
        if not np.array_equal(d["test_rally_uid"], ref_test_uid):
            print(f"   WARN: test_rally_uid differs from reference")
        else:
            print(f"   OK alignment: y_act/y_pt match; test_uid match")

        # Action class handling
        act_shape = d["oof_act"].shape
        test_act_shape = d["test_act"].shape
        print(f"   action shapes: oof={act_shape}  test={test_act_shape}")

        teacher_data[tag] = d

    if len(teacher_data) < 2:
        print("\n FATAL: need at least 2 teachers to compute correlations.")
        sys.exit(1)

    # ─── Correlation: action ─────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(" ACTION correlations (sliced to first 15 classes)")
    print("=" * 78)
    act_probs = {t: slice_to_15_classes(d["oof_act"]) for t, d in teacher_data.items()}
    test_act_probs = {t: slice_to_15_classes(d["test_act"]) for t, d in teacher_data.items()}

    print("\n  OOF top-1 agreement matrix (fraction of rows with matching argmax):")
    top1_corr = pairwise_top1_correlation(act_probs)
    print(top1_corr.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n  OOF probability Pearson correlation matrix:")
    prob_corr = pairwise_prob_correlation(act_probs)
    print(prob_corr.to_string(float_format=lambda x: f"{x:.4f}"))

    # ─── Correlation: point ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(" POINT correlations (full 10 classes)")
    print("=" * 78)
    pt_probs = {t: d["oof_pt"].astype(np.float32) for t, d in teacher_data.items()}
    print("\n  OOF top-1 agreement matrix:")
    pt_top1_corr = pairwise_top1_correlation(pt_probs)
    print(pt_top1_corr.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n  OOF probability Pearson correlation matrix:")
    pt_prob_corr = pairwise_prob_correlation(pt_probs)
    print(pt_prob_corr.to_string(float_format=lambda x: f"{x:.4f}"))

    # ─── Correlation: server ─────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(" SERVER (binary) correlations")
    print("=" * 78)
    srv = {t: d["oof_srv"].astype(np.float64) for t, d in teacher_data.items()}
    tags = list(srv.keys())
    n = len(tags)
    srv_corr = np.zeros((n, n))
    for i, ta in enumerate(tags):
        for j, tb in enumerate(tags):
            srv_corr[i, j] = float(np.corrcoef(srv[ta], srv[tb])[0, 1])
    srv_df = pd.DataFrame(srv_corr, index=tags, columns=tags)
    print(srv_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # ─── Cluster verdict ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f" CLUSTER REPORT (correlation block threshold {CORRELATION_BLOCK_THRESHOLD})")
    print("=" * 78)
    print("\n Action top-1-agreement clusters:")
    act_clusters = cluster_report(top1_corr, CORRELATION_BLOCK_THRESHOLD)
    for i, cl in enumerate(act_clusters, 1):
        print(f"  Cluster {i}: {cl}")

    high_corr_pairs = []
    for i, ta in enumerate(top1_corr.index):
        for j, tb in enumerate(top1_corr.columns):
            if j <= i:
                continue
            if float(top1_corr.iloc[i, j]) >= CORRELATION_BLOCK_THRESHOLD:
                high_corr_pairs.append((ta, tb, float(top1_corr.iloc[i, j])))
    print(f"\n Action high-corr pairs (>= {CORRELATION_BLOCK_THRESHOLD}):")
    for a, b, c in high_corr_pairs:
        print(f"   {a} <-> {b}  corr={c:.4f}")
    if not high_corr_pairs:
        print("   (none — design's decorrelation gate would PASS)")

    # ─── Manifest ────────────────────────────────────────────────────────────
    manifest = {
        "stage": "R-065 Stage-0 audit",
        "ts": "2026-05-23",
        "canonical_oof_size": int(n_canonical),
        "test_uid_count": int(len(ref_test_uid)),
        "test_uid_sha256_16": uid_hash,
        "correlation_block_threshold": CORRELATION_BLOCK_THRESHOLD,
        "teachers_loaded": list(teacher_data.keys()),
        "teachers_missing": [t for t, _ in TEACHERS if t not in teacher_data],
        "action_top1_corr": top1_corr.to_dict(),
        "action_prob_corr": prob_corr.to_dict(),
        "point_top1_corr": pt_top1_corr.to_dict(),
        "point_prob_corr": pt_prob_corr.to_dict(),
        "srv_corr": srv_df.to_dict(),
        "action_clusters": act_clusters,
        "high_corr_pairs_action_top1": [
            {"a": a, "b": b, "corr": c} for a, b, c in high_corr_pairs
        ],
        "decorrelation_gate_pass": len(high_corr_pairs) == 0,
    }
    out_path = os.path.join(PROJECT_ROOT, "submissions", "r065_teacher_correlation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, default=float)
    print(f"\n Manifest saved: {out_path}")

    if not manifest["decorrelation_gate_pass"]:
        print("\n VERDICT: decorrelation gate FAILS — Consensus Pseudo V2 with this teacher set is BLOCKED.")
        print("          Recommend: drop highest-corr teacher, find a replacement, or rerun audit.")
    else:
        print("\n VERDICT: decorrelation gate PASSES — Stage-0 generator may proceed.")


if __name__ == "__main__":
    main()
