"""Diversity / correlation audit across all 69712-row OOF components.

Goal: find where R-067cr blend is missing diversity. Reveal candidate
mechanisms by identifying:
  1. Highly-correlated component pairs (redundant)
  2. Components with unique signal (high info value)
  3. Rows where ALL components are uncertain (improvement frontier)
  4. Per-class which components are strongest (specialist signal)

USAGE:
    python -u src/diversity_audit_2026-05-26.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, SUBMISSION_DIR

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
N_ACTION_TRAIN = 15
N_POINT = 10
N_ACTION_FULL = 19


def pad19(arr):
    if arr.shape[1] >= N_ACTION_FULL:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((arr.shape[0], N_ACTION_FULL), dtype=np.float32)
    out[:, :arr.shape[1]] = arr
    return out


def load_component_aligned(tag):
    """Load + slice to 69712 if needed."""
    base = os.path.join(OOF_DIR, tag)
    for s in ["oof_act", "oof_pt", "oof_srv"]:
        if not os.path.exists(f"{base}_{s}.npy"):
            return None
    oa = pad19(np.load(f"{base}_oof_act.npy"))[:69712]
    op = np.load(f"{base}_oof_pt.npy").astype(np.float32)[:69712]
    os_ = np.load(f"{base}_oof_srv.npy").astype(np.float32)[:69712]
    if oa.shape[0] != 69712:
        return None
    return {"oof_act": oa, "oof_pt": op, "oof_srv": os_}


COMPONENTS_OF_INTEREST = [
    # R-034 PAIR (current LB-best)
    "v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3",
    # V11 family extras
    "v11", "v11_aug", "v11_aug_big",
    # V14 family
    "v14_seed2",
    # Specialized
    "v11_mulminet_aug_oldtest_softf1_phaseB",
    "sgp_prefix_v3_full",
    # Meta (LB-failed but instructive for diversity audit)
    "meta_stack", "meta_stack_v2_logistic",
]


def main():
    print("=" * 80)
    print(" Diversity / correlation audit across OOF components")
    print("=" * 80)

    # Reference labels from v14_seed2_v15feat_a (canonical 69712-aligned)
    REF = "v14_seed2_v15feat_a"
    y_act = np.load(os.path.join(OOF_DIR, f"{REF}_oof_y_act.npy"))
    y_pt  = np.load(os.path.join(OOF_DIR, f"{REF}_oof_y_pt.npy"))
    y_srv = np.load(os.path.join(OOF_DIR, f"{REF}_oof_y_srv.npy"))
    y_act_clip = np.where(y_act >= N_ACTION_TRAIN, 0, y_act)

    # Load components that align
    comp_data = {}
    for tag in COMPONENTS_OF_INTEREST:
        d = load_component_aligned(tag)
        if d is not None:
            comp_data[tag] = d
            print(f"  loaded {tag:<40} act{d['oof_act'].shape}")
        else:
            print(f"  SKIP   {tag:<40} (alignment/missing)")
    n_comp = len(comp_data)

    # ─── 1. Standalone F1 / AUC per task ────────────────────────────────
    print("\n" + "=" * 80)
    print(" 1. Standalone metric per component")
    print("=" * 80)
    print(f"{'component':<42} {'F1_a':>8} {'F1_p':>8} {'AUC':>8} {'OV':>8}")
    standalone = {}
    for tag, d in comp_data.items():
        pa = d["oof_act"][:, :N_ACTION_TRAIN].argmax(axis=1)
        pp = d["oof_pt"].argmax(axis=1)
        f1a = f1_score(y_act_clip, pa, labels=list(range(N_ACTION_TRAIN)),
                        average="macro", zero_division=0)
        f1p = f1_score(y_pt, pp, labels=list(range(N_POINT)),
                        average="macro", zero_division=0)
        srv_m = (y_srv >= 0)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_srv[srv_m], d["oof_srv"][srv_m])
        except Exception:
            auc = 0.5
        ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
        standalone[tag] = {"F1_a": float(f1a), "F1_p": float(f1p),
                            "AUC": float(auc), "OV": float(ov)}
        print(f"{tag:<42} {f1a:.4f}   {f1p:.4f}   {auc:.4f}   {ov:.4f}")

    # ─── 2. Top-1 agreement matrix (action task) ────────────────────────
    print("\n" + "=" * 80)
    print(" 2. Action top-1 agreement matrix (1.0 = same prediction; <0.85 = diverse)")
    print("=" * 80)
    tags = list(comp_data.keys())
    n = len(tags)
    argmaxes = {tag: comp_data[tag]["oof_act"][:, :N_ACTION_TRAIN].argmax(axis=1) for tag in tags}
    agree_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            agree_mat[i, j] = (argmaxes[tags[i]] == argmaxes[tags[j]]).mean()
    # Print compact header (use short labels)
    short = {t: t.replace("v11_", "v11_").replace("v14_", "v14_")[:10] for t in tags}
    print("  " + "  ".join(f"{short[t][:9]:>9}" for t in tags[:n_comp]))
    for i in range(n):
        row = " ".join(f"{agree_mat[i, j]:>9.3f}" for j in range(n))
        print(f"{short[tags[i]][:8]:>8} {row}")

    # Find pairs with highest disagreement (lowest agreement)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((tags[i], tags[j], float(agree_mat[i, j])))
    pairs.sort(key=lambda x: x[2])
    print("\n   Top-10 most diverse pairs (lowest agreement = max diversity):")
    for t1, t2, a in pairs[:10]:
        print(f"     {t1:<35} vs {t2:<35} = {a:.4f}")

    # ─── 3. Per-class action F1 — find specialists ──────────────────────
    print("\n" + "=" * 80)
    print(" 3. Per-class action F1 — which component is best at each class?")
    print("=" * 80)
    class_names = ["None", "Loop", "Cloop", "Smash", "Flip", "Pushfast", "Push",
                    "Flick", "Arch", "Knuckle", "Chop_r", "ShortStop", "Chop",
                    "Block", "Lob"]
    per_class = {}
    for tag, d in comp_data.items():
        pred = d["oof_act"][:, :N_ACTION_TRAIN].argmax(axis=1)
        f1s = f1_score(y_act_clip, pred, labels=list(range(N_ACTION_TRAIN)),
                        average=None, zero_division=0)
        per_class[tag] = f1s
    print(f"{'class':<14} " + "  ".join(f"{tag[:10]:>10}" for tag in tags))
    for c in range(N_ACTION_TRAIN):
        row = "  ".join(f"{per_class[tag][c]:>10.4f}" for tag in tags)
        # Mark best per class
        vals = np.array([per_class[tag][c] for tag in tags])
        best_idx = vals.argmax()
        print(f"  act{c:>2} {class_names[c][:6]:<6} {row}  best={tags[best_idx][:20]}")

    # ─── 4. "Hard" rows: where ALL R-034 components have low max-prob ───
    print("\n" + "=" * 80)
    print(" 4. Hardest 50 rows (lowest max-prob across R-034 components)")
    print("=" * 80)
    R034 = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3"]
    if all(t in comp_data for t in R034):
        max_probs_act = np.stack([comp_data[t]["oof_act"][:, :N_ACTION_TRAIN].max(axis=1)
                                    for t in R034], axis=0)
        min_max_prob_per_row = max_probs_act.min(axis=0)  # weakest component's confidence
        avg_max_prob_per_row = max_probs_act.mean(axis=0)
        hardest = np.argsort(avg_max_prob_per_row)[:50]
        print(f"  median avg-max-prob across all rows: {np.median(avg_max_prob_per_row):.4f}")
        print(f"  hardest 50 rows median: {np.median(avg_max_prob_per_row[hardest]):.4f}")
        print(f"  hardest 50 rows mean true class:")
        from collections import Counter
        cnt = Counter(y_act_clip[hardest].tolist())
        for cls, n in cnt.most_common(5):
            print(f"     act{cls} ({class_names[cls]}): {n} rows")
        # Are the hardest rows in any particular SN bucket?
        nsn = np.load(os.path.join(OOF_DIR, f"{REF}_oof_nsn.npy"))
        nsn_hardest = nsn[hardest]
        print(f"  hardest 50 rows SN distribution: SN<=2={int((nsn_hardest<=2).sum())}  "
              f"SN 3-4={int(((nsn_hardest>=3)&(nsn_hardest<=4)).sum())}  "
              f"SN>=5={int((nsn_hardest>=5).sum())}")

    # ─── 5. SAVE manifest ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" Saving manifest")
    print("=" * 80)
    manifest = {
        "ts": "2026-05-26",
        "components_analyzed": list(comp_data.keys()),
        "standalone_metrics": standalone,
        "action_argmax_agreement_matrix": agree_mat.tolist(),
        "tags_order": tags,
        "most_diverse_pairs": [
            {"comp1": p[0], "comp2": p[1], "agreement": p[2]} for p in pairs[:20]
        ],
        "per_class_action_f1": {tag: per_class[tag].tolist() for tag in tags},
    }
    out = os.path.join(SUBMISSION_DIR, "diversity_audit_2026-05-26.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f" Saved: {out}")


if __name__ == "__main__":
    main()
