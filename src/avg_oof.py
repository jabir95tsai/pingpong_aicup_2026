"""Average OOF probability arrays across multiple seeds and blend with V11.

Averages raw probability arrays (not argmax labels) from multiple seed runs,
then pipes the averaged GBM output through blend_ensemble logic to blend with
the V11 transformer.

Usage (3-seed average then blend):
    python src/avg_oof.py \
        --tags v14_seed0 v14_seed1 v14_seed2 \
        --out-tag v14_avg3 \
        --blend-v11

Output:
    oof_predictions/v14_avg3_oof_act.npy   (averaged action probs)
    oof_predictions/v14_avg3_oof_pt.npy    (averaged point probs)
    oof_predictions/v14_avg3_oof_srv.npy   (averaged server probs)
    oof_predictions/v14_avg3_oof_mask.npy  (union mask — must equal first tag's mask)
    oof_predictions/v14_avg3_oof_y_act.npy (copied from first tag)
    oof_predictions/v14_avg3_oof_y_pt.npy
    oof_predictions/v14_avg3_oof_y_srv.npy
    oof_predictions/v14_avg3_oof_nsn.npy
    oof_predictions/v14_avg3_test_act.npy  (averaged test probs)
    oof_predictions/v14_avg3_test_pt.npy
    oof_predictions/v14_avg3_test_srv.npy
    oof_predictions/v14_avg3_test_rally_uid.npy (from first tag)
"""
import argparse, os, sys
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SUBMISSION_DIR

N_ACTION       = 19
N_ACTION_TRAIN = 15
N_POINT        = 10
ACTION_EVAL_LABELS = list(range(15))
POINT_EVAL_LABELS  = list(range(10))


def apply_action_rules(probs, next_sns):
    out = probs.copy()
    serve_mask = (next_sns == 1)
    non_serve  = ~serve_mask
    out[serve_mask, :15] = 0.0
    for c in [15, 16, 17, 18]:
        if c < out.shape[1]:
            out[non_serve, c] = 0.0
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return out / row_sums


def load_tag(oof_dir, tag):
    def p(suffix):
        return os.path.join(oof_dir, f"{tag}_{suffix}.npy")
    return {
        "oof_act":       np.load(p("oof_act")),
        "oof_pt":        np.load(p("oof_pt")),
        "oof_srv":       np.load(p("oof_srv")),
        "oof_mask":      np.load(p("oof_mask")),
        "oof_y_act":     np.load(p("oof_y_act")),
        "oof_y_pt":      np.load(p("oof_y_pt")),
        "oof_y_srv":     np.load(p("oof_y_srv")),
        "oof_nsn":       np.load(p("oof_nsn")),
        "test_act":      np.load(p("test_act")),
        "test_pt":       np.load(p("test_pt")),
        "test_srv":      np.load(p("test_srv")),
        "test_rally_uid": np.load(p("test_rally_uid")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags",     nargs="+", required=True,
                        help="Space-separated list of seed run tags to average")
    parser.add_argument("--out-tag",  type=str, required=True,
                        help="Output tag for averaged artifacts")
    parser.add_argument("--blend-v11", action="store_true",
                        help="After averaging, run blend_ensemble with V11")
    parser.add_argument("--out",      type=str, default=None,
                        help="Submission filename for blend (default: submission_<out-tag>_v11_optblend.csv)")
    args = parser.parse_args()

    oof_dir = os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions")

    print(f"=== Averaging {len(args.tags)} seed runs: {args.tags} ===")
    runs = [load_tag(oof_dir, t) for t in args.tags]

    # Validate consistency
    ref = runs[0]
    for i, r in enumerate(runs[1:], 1):
        assert np.array_equal(r["oof_mask"], ref["oof_mask"]), \
            f"FAIL: oof_mask mismatch between {args.tags[0]} and {args.tags[i]}"
        assert np.array_equal(r["oof_y_act"], ref["oof_y_act"]), \
            f"FAIL: oof_y_act mismatch between {args.tags[0]} and {args.tags[i]}"
        assert np.array_equal(r["test_rally_uid"], ref["test_rally_uid"]), \
            f"FAIL: test_rally_uid mismatch between {args.tags[0]} and {args.tags[i]}"
    print(f"  Consistency checks passed ({len(runs)} runs)")

    # Average probability arrays
    avg_oof_act = np.mean([r["oof_act"] for r in runs], axis=0)
    avg_oof_pt  = np.mean([r["oof_pt"]  for r in runs], axis=0)
    avg_oof_srv = np.mean([r["oof_srv"] for r in runs], axis=0)
    avg_test_act = np.mean([r["test_act"] for r in runs], axis=0)
    avg_test_pt  = np.mean([r["test_pt"]  for r in runs], axis=0)
    avg_test_srv = np.mean([r["test_srv"] for r in runs], axis=0)

    mask     = ref["oof_mask"]
    y_act    = ref["oof_y_act"]
    y_pt     = ref["oof_y_pt"]
    y_srv    = ref["oof_y_srv"]
    nsn      = ref["oof_nsn"]

    # Evaluate averaged OOF
    act_ruled = apply_action_rules(avg_oof_act[mask], nsn[mask])
    f1_a = f1_score(y_act[mask], np.argmax(act_ruled, axis=1),
                    labels=ACTION_EVAL_LABELS, average="macro", zero_division=0)
    f1_p = f1_score(y_pt[mask],  np.argmax(avg_oof_pt[mask], axis=1),
                    labels=POINT_EVAL_LABELS, average="macro", zero_division=0)
    auc  = roc_auc_score(y_srv[mask], avg_oof_srv[mask])
    ov   = 0.4*f1_a + 0.4*f1_p + 0.2*auc
    print(f"  Averaged OOF: action={f1_a:.4f}  point={f1_p:.4f}  AUC={auc:.4f}  OV={ov:.4f}")
    print(f"  OOF mask: {mask.sum()}/{len(mask)}")

    # Save averaged artifacts
    out_tag = args.out_tag
    def sp(suffix): return os.path.join(oof_dir, f"{out_tag}_{suffix}.npy")
    np.save(sp("oof_act"),        avg_oof_act)
    np.save(sp("oof_pt"),         avg_oof_pt)
    np.save(sp("oof_srv"),        avg_oof_srv)
    np.save(sp("oof_mask"),       mask)
    np.save(sp("oof_y_act"),      y_act)
    np.save(sp("oof_y_pt"),       y_pt)
    np.save(sp("oof_y_srv"),      y_srv)
    np.save(sp("oof_nsn"),        nsn)
    np.save(sp("test_act"),       avg_test_act)
    np.save(sp("test_pt"),        avg_test_pt)
    np.save(sp("test_srv"),       avg_test_srv)
    np.save(sp("test_rally_uid"), ref["test_rally_uid"])
    print(f"  Saved averaged artifacts → {oof_dir}/{out_tag}_*.npy")

    if args.blend_v11:
        import subprocess, sys
        out_csv = args.out or f"submission_{out_tag}_v11_optblend.csv"
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "blend_ensemble.py"),
               "--v1", out_tag, "--aux-tag", "v11", "--out", out_csv]
        print(f"\n=== Running blend: {' '.join(cmd)} ===")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
