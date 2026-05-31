"""R-067cr gap reassessment — where does the +0.0130 LB gap to 0.4000 live?

Reads the R-034 PAIR blend (R-067cr's action/point base) on OOF, reproduces
the LB-best per-task Dirichlet weights, and decomposes macro-F1 per class to
find the highest-leverage targets. Pure analysis; writes a markdown report.

Macro-F1 = mean over eval classes. With 15 action classes, lifting ONE dead
class from F1=0 to F1=0.30 adds 0.30/15 = 0.020 to F1_a, i.e. +0.008 OV
(0.4 weight). For point (10 classes): 0.30/10 = 0.030 -> +0.012 OV. So the
single biggest leverage is a currently-dead POINT class.
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_oldtest_blend import (  # noqa: E402
    load_components, evaluate_subset_none,
    ACTION_EVAL, POINT_EVAL, N_ACTION, N_POINT,
)

R034_SUBSET = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3"]

ACTION_NAMES = {
    0: "none", 1: "loop", 2: "counter-loop", 3: "smash", 4: "twist", 5: "fast-drive",
    6: "push-block", 7: "flick", 8: "arc/hook", 9: "tap", 10: "chop-push",
    11: "short-stop/short-chop", 12: "chop", 13: "block", 14: "lob",
}
POINT_NAMES = {
    0: "miss/net", 1: "FH-short", 2: "mid-short", 3: "BH-short",
    4: "FH-half", 5: "mid-half", 6: "BH-half", 7: "FH-long", 8: "mid-long", 9: "BH-long",
}


def per_class_f1(y_true, y_pred, labels, n_total):
    cm = np.bincount(y_true.astype(np.int64) * n_total + y_pred.astype(np.int64),
                     minlength=n_total * n_total).reshape(n_total, n_total)
    col_sum = cm.sum(axis=0)
    row_sum = cm.sum(axis=1)
    diag = np.diag(cm)
    out = []
    for c in labels:
        tp = diag[c]; fp = col_sum[c] - tp; fn = row_sum[c] - tp
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom <= 0 else (2 * tp) / denom
        prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        out.append((c, f1, prec, rec, int(row_sum[c]), int(col_sum[c])))
    return out, cm


def top_confusions(cm, c, names, k=3):
    row = cm[c].copy()
    row[c] = 0
    idx = np.argsort(-row)[:k]
    return [(int(j), names.get(int(j), str(j)), int(row[j])) for j in idx if row[j] > 0]


def main():
    print("Loading R-034 components + optimizing Dirichlet weights ...")
    comp, y_a, y_p, y_s, mask, _ = load_components(R034_SUBSET)
    res = evaluate_subset_none(R034_SUBSET, comp, y_a, y_p, y_s,
                               optimize=True, n_samples=400)
    w_a, w_p = res["w_a"], res["w_p"]
    print(f"  OV={res['OV']:.4f}  F1_a={res['F1_a']:.4f}  F1_p={res['F1_p']:.4f}  AUC={res['AUC']:.4f}")

    act_stack = np.stack([comp[t]["oof_act"] for t in R034_SUBSET], axis=0)
    pt_stack = np.stack([comp[t]["oof_pt"] for t in R034_SUBSET], axis=0)
    blend_a = (w_a[:, None, None] * act_stack).sum(axis=0).argmax(axis=1)
    blend_p = (w_p[:, None, None] * pt_stack).sum(axis=0).argmax(axis=1)

    af, acm = per_class_f1(y_a, blend_a, ACTION_EVAL, N_ACTION)
    pf, pcm = per_class_f1(y_p, blend_p, POINT_EVAL, N_POINT)

    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 78)
    emit(" R-067cr GAP REASSESSMENT — per-class macro-F1 decomposition (OOF)")
    emit("=" * 78)
    emit(f" Blend OV={res['OV']:.4f}  F1_a={res['F1_a']:.4f}  F1_p={res['F1_p']:.4f}  AUC={res['AUC']:.4f}")
    emit(f" LB anchor (R-067cr) = 0.3870095   target = 0.4000   gap = +0.0130")
    emit("")

    for title, feats, names, ncls in [
        ("ACTION (weight 0.4, 15 eval classes)", af, ACTION_NAMES, 15),
        ("POINT  (weight 0.4, 10 eval classes)", pf, POINT_NAMES, 10),
    ]:
        emit("-" * 78)
        emit(f" {title}")
        emit("-" * 78)
        emit(f" {'cls':>3} {'name':<22} {'F1':>6} {'prec':>6} {'rec':>6} {'supp':>6} {'pred':>6}  topConfusions")
        for (c, f1, prec, rec, supp, predN) in sorted(feats, key=lambda r: r[1]):
            conf = top_confusions(acm if ncls == 15 else pcm, c, names)
            confs = ", ".join(f"{nm}:{n}" for _, nm, n in conf)
            emit(f" {c:>3} {names.get(c,'?'):<22} {f1:>6.3f} {prec:>6.3f} {rec:>6.3f} {supp:>6} {predN:>6}  {confs}")
        # headroom: lift each class to 0.30 (cap at current+0.30), report OV delta
        emit("")
        emit(f"  HEADROOM (lift each weak class -> F1=0.30, OV impact = 0.4 * dF1/{ncls}):")
        for (c, f1, prec, rec, supp, predN) in sorted(feats, key=lambda r: r[1]):
            if f1 < 0.30 and supp >= 30:
                d_f1 = (0.30 - f1) / ncls
                d_ov = 0.4 * d_f1
                emit(f"    cls {c:>2} {names.get(c,'?'):<22} F1 {f1:.3f}->0.30  +{d_f1:.4f} F1  => +{d_ov:.4f} OV  (supp={supp})")
        emit("")

    # Summary: cumulative headroom if all sub-0.30 classes (supp>=30) reach 0.30
    def cum(feats, ncls):
        tot = 0.0
        for (c, f1, _, _, supp, _) in feats:
            if f1 < 0.30 and supp >= 30:
                tot += 0.4 * (0.30 - f1) / ncls
        return tot
    ch_a = cum(af, 15); ch_p = cum(pf, 10)
    emit("=" * 78)
    emit(" CUMULATIVE OV HEADROOM (all sub-0.30, supp>=30 classes -> 0.30)")
    emit("=" * 78)
    emit(f"   action: +{ch_a:.4f} OV    point: +{ch_p:.4f} OV    total: +{ch_a+ch_p:.4f} OV")
    emit(f"   (gap to 0.4000 is +0.0130; realistic capture is a fraction of this ceiling)")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "audits", "R067cr_gap_reassessment_2026-05-29.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# R-067cr Gap Reassessment (auto-generated)\n\n```\n")
        f.write("\n".join(lines))
        f.write("\n```\n")
    print(f"\n report -> {out_path}")


if __name__ == "__main__":
    main()
